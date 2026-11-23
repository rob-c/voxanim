"""The experiment behind animation 178, ``from-a-cut-to-a-tree``: the bridge from Part 1's
selections and tests ("Moving a cut", "The likelihood ratio is the best test") to Part 2's
decision trees ("Growing a decision tree", "Choosing a split"); see
:func:`voxanim.next_catalog.redesigned`.

The events are those of :mod:`voxanim.study_trees`: MNIST 4s stand for signal and 9s for
background, each read as two measured features, the ink at the top centre of the picture
and how far the ink is spread up and down. The 4,000 of each in its training pool are the
training events here; the other 5,782 4s and 9s (the rest of the training file and the
whole test file) are held out, and every efficiency, background share and area quoted is
measured on them unless it says otherwise.

* One feature, every threshold: an analysis keeps the events on the signal side of a cut
  and chooses the cut by the figure of merit ``s / sqrt(s + b)`` of the training events it
  keeps (:func:`significance`); a tree's root chooses it by the Gini gain of the split
  (:func:`voxanim.study_trees.split_gain`). Both are scored at every midpoint between
  neighbouring training values, and the two chosen cuts compared, as counted and with the
  4s made rare (50 of them for every 1,000 9s, as in "Moving a cut").
* The cut chosen by Gini is the root of a tree: a stump, a tree of depth one. Asking again
  in each half, then again, gives the tree of :func:`voxanim.study_trees.grow`, its
  leaves boxes in the plane of the two features.
* ROC curves of the held-out events (:func:`roc`): the one feature cut at every threshold,
  the tree cut at each depth (an event scored by its leaf's share of training 4s), and
  the likelihood ratio estimated from two 2-D histograms of the training events
  (:func:`ratio_table`), the Neyman-Pearson best test when the histograms are right.

Plain numpy, no random numbers beyond the seeded division of events made by
:func:`voxanim.study_trees.events` (seed recorded).
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import study_trees as T
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
NAME = "from-a-cut-to-a-tree"
EVIDENCE = dict(
    exact="exact arithmetic on the training events: counts kept by a cut, s / sqrt(s + b) "
          "and Gini gains worked out from them",
    measured="measured on MNIST: a tree and two histograms built here in numpy from 8,000 "
             "training events, scored on 5,782 held-out events",
)
FEATURE = 1                       # the one feature cut first: the ink's vertical spread
RARE = (50.0, 1000.0)             # an analysis with rare signal: 50 4s per 1,000 9s
RATIO_BINS = 10                   # equal-count bins of each feature for the 2-D histograms
RATIO_SWEEP = (4, 6, 8, 10, 12, 16, 20, 30, 40)
SMOOTH = 0.5                      # added to every histogram cell before the ratio
GRID = 201                        # signal efficiencies at which the ROC curves are kept
PLANE_EVENTS = 300                # training events of each class kept for the plane
HIST_BINS = 32                    # bins of the one feature's histograms, over the plane


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-cut-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== one cut
def significance(s, b):
    """``s / sqrt(s + b)``: how many standard deviations the kept signal ``s`` stands
    above the fluctuation of all that is kept (0 where nothing is kept)."""
    s, b = np.asarray(s, float), np.asarray(b, float)
    return np.where(s + b > 0, s / np.sqrt(np.maximum(s + b, 1e-300)), 0.0)


def kept(x, y, cuts, below=True):
    """The signal and background events kept by each threshold of ``cuts``: those with
    ``x <= cut`` (``below``) or ``x > cut``. Returns ``(s, b)``."""
    x, y = np.asarray(x, float), np.asarray(y)
    xs, xb = np.sort(x[y == 1]), np.sort(x[y == 0])
    s = np.searchsorted(xs, cuts, side="right").astype(float)
    b = np.searchsorted(xb, cuts, side="right").astype(float)
    if not below:
        s, b = len(xs) - s, len(xb) - b
    return s, b


def signal_side(x, y):
    """True when signal sits below background on this feature (by their medians): the
    side an analysis keeps."""
    x, y = np.asarray(x, float), np.asarray(y)
    return bool(np.median(x[y == 1]) < np.median(x[y == 0]))


def scan(x, y, weights=(1.0, 1.0)):
    """One feature, every threshold: the midpoints between neighbouring distinct values,
    the signal and background events kept on the signal side of each, ``s / sqrt(s + b)``
    of them and the Gini gain of the split. ``weights``: what one signal and one
    background event count for (a rare signal is a small weight). Returns a dict."""
    x, y = np.asarray(x, float), np.asarray(y)
    v = np.unique(x)
    cuts = (v[:-1] + v[1:]) / 2
    below = signal_side(x, y)
    s, b = kept(x, y, cuts, below)
    ws, wb = weights
    S, B = float((y == 1).sum()), float((y == 0).sum())
    l1, l0 = kept(x, y, cuts, True)                      # the split's left side: x <= cut
    gain = T.split_gain(ws * S + wb * B, ws * S, ws * l1 + wb * l0, ws * l1, "gini")
    fom = significance(ws * s, wb * b)
    return dict(cuts=cuts, s=s, b=b, fom=fom, gain=gain, below=below,
                fom_best=int(np.argmax(fom)), gain_best=int(np.argmax(gain)))


# ====================================================================== ROC and ratio
def roc(score_s, score_b):
    """The ROC curve of a score that is larger for signal: from keeping nothing to
    keeping everything, one corner for each distinct score (events with equal scores are
    kept together, so between corners the curve is straight). Returns ``(eff, kept)``:
    the shares of signal and of background kept, both rising from 0 to 1."""
    ss, sb = np.asarray(score_s, float), np.asarray(score_b, float)
    v = np.unique(np.r_[ss, sb])[::-1]
    cs = np.bincount(len(v) - 1 - np.searchsorted(v[::-1], ss), minlength=len(v))
    cb = np.bincount(len(v) - 1 - np.searchsorted(v[::-1], sb), minlength=len(v))
    eff = np.r_[0.0, np.cumsum(cs) / len(ss)]
    kept_b = np.r_[0.0, np.cumsum(cb) / len(sb)]
    return eff, kept_b


def area(eff, kept_b):
    """The area under the ROC curve drawn as background rejection against efficiency:
    the chance that a random signal event outscores a random background one (ties
    counting half)."""
    eff, kept_b = np.asarray(eff, float), np.asarray(kept_b, float)
    return float(np.sum(np.diff(kept_b) * (eff[1:] + eff[:-1]) / 2))


def at_efficiency(eff, kept_b, e):
    """The background kept at signal efficiency ``e``, along the straight pieces of the
    curve (where a vertical piece passes through ``e``, the lowest point of it)."""
    eff, kept_b = np.asarray(eff, float), np.asarray(kept_b, float)
    order = np.lexsort((kept_b, eff))
    eff, kept_b = eff[order], kept_b[order]
    ue, first = np.unique(eff, return_index=True)
    last = np.r_[first[1:], len(eff)] - 1
    lo, hi = kept_b[first], kept_b[last]             # the bottom and top of each vertical
    e = np.asarray(e, float)
    j = np.clip(np.searchsorted(ue, e, side="right") - 1, 0, len(ue) - 1)
    nxt = np.minimum(j + 1, len(ue) - 1)
    span = np.where(nxt > j, ue[nxt] - ue[j], 1.0)
    f = np.clip((e - ue[j]) / span, 0.0, 1.0)
    out = np.where(e <= ue[j], lo[j], hi[j] + f * (lo[nxt] - hi[j]))
    return out if out.ndim else float(out)


def ratio_edges(x, bins):
    """Equal-count bin edges of each feature of ``x``: ``bins - 1`` inner edges each."""
    x = np.asarray(x, float)
    q = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.array([np.quantile(x[:, f], q) for f in range(x.shape[1])])


def ratio_cells(x, edges):
    """The 2-D histogram cell ``(row, col)`` of each event: row from the first feature."""
    x = np.asarray(x, float)
    return tuple(np.searchsorted(edges[f], x[:, f], side="right") for f in range(2))


def ratio_table(x, y, bins, smooth=SMOOTH):
    """The likelihood ratio estimated from two 2-D histograms of the training events:
    in each cell, the share of all signal events that fall there over the share of all
    background events, ``smooth`` added to each count first so that no cell is 0 or
    infinite. Returns ``(edges, ratio, signal counts, background counts)``."""
    x, y = np.asarray(x, float), np.asarray(y)
    edges = ratio_edges(x, bins)
    hs, hb = np.zeros((bins, bins)), np.zeros((bins, bins))
    r, c = ratio_cells(x, edges)
    np.add.at(hs, (r[y == 1], c[y == 1]), 1.0)
    np.add.at(hb, (r[y == 0], c[y == 0]), 1.0)
    S, B = float((y == 1).sum()), float((y == 0).sum())
    ratio = ((hs + smooth) / (S + smooth * bins * bins)) / ((hb + smooth) / (B + smooth
                                                                           * bins * bins))
    return edges, ratio, hs, hb


def ratio_score(x, edges, ratio):
    """Each event's estimated likelihood ratio: its cell's."""
    return ratio[ratio_cells(x, edges)]


# ====================================================================== the run
def data():
    """``(x, y, hx, hy)``: the two features and labels (1: a 4) of the training pool and
    of the held-out events."""
    e = T.events()
    return e["pool_two"], e["pool_y"], e["held_two"], e["held_y"]


def _build():
    x, y, hx, hy = data()
    out = {}
    for f in range(2):
        for tag, w in (("", (1.0, 1.0)),
                       ("rare_", (RARE[0] / (y == 1).sum(), RARE[1] / (y == 0).sum()))):
            sc = scan(x[:, f], y, w)
            for k in ("cuts", "s", "b", "fom", "gain"):
                out["%sscan%d_%s" % (tag, f, k)] = sc[k]
            out["%sscan%d_best" % (tag, f)] = np.array([sc["fom_best"], sc["gain_best"]])
            out["%sscan%d_below" % (tag, f)] = np.array(sc["below"])
    tree, cuts = T.fit_two(x, y)
    for k, v in tree.items():
        if k != "tried":
            out["tree_" + k] = v
    out["tree_cuts"] = cuts
    xb, hxb = T.binned(x, cuts), T.binned(hx, cuts)
    # the stump's own corner on held-out events: where the curves are compared
    root = float(cuts[tree["feature"][0], tree["cut"][0]])
    on = (hx[:, tree["feature"][0]] <= root) == bool(out["scan%d_below" % FEATURE])
    out["stump_held"] = np.array([on[hy == 1].mean(), on[hy == 0].mean()])
    marks = np.array([out["stump_held"][0], 0.8])     # efficiencies the curves are read at
    deepest = int(tree["depth"].max())
    grid = np.linspace(0.0, 1.0, GRID)
    rows, curves = [], []
    for d in range(1, deepest + 1):
        st, sh = T.share(tree, xb, max_depth=d), T.share(tree, hxb, max_depth=d)
        e_t, k_t = roc(st[y == 1], st[y == 0])
        e_h, k_h = roc(sh[hy == 1], sh[hy == 0])
        rows.append([d, T.leaf_count(tree, max_depth=d), area(e_t, k_t), area(e_h, k_h),
                     *at_efficiency(e_h, k_h, marks)])
        curves.append(at_efficiency(e_h, k_h, grid))
    out["depths"] = np.array(rows)
    out["depth_curves"] = np.array(curves)
    out["grid"] = grid
    # the one feature, cut at every threshold: the score is the feature, signal side up
    sign = 1.0 if not out["scan%d_below" % FEATURE] else -1.0
    e_c, k_c = roc(sign * hx[hy == 1, FEATURE], sign * hx[hy == 0, FEATURE])
    out["cut_curve"] = at_efficiency(e_c, k_c, grid)
    out["cut_area"] = np.array(area(e_c, k_c))
    out["cut_at"] = at_efficiency(e_c, k_c, marks)
    # the likelihood ratio from histograms of the training events
    sweep = []
    for bins in RATIO_SWEEP:
        edges, ratio, _, _ = ratio_table(x, y, bins)
        r_h = ratio_score(hx, edges, ratio)
        sweep.append(area(*roc(r_h[hy == 1], r_h[hy == 0])))
    out["ratio_sweep"] = np.array(sweep)
    edges, ratio, hs, hb = ratio_table(x, y, RATIO_BINS)
    r_h, r_t = ratio_score(hx, edges, ratio), ratio_score(x, edges, ratio)
    e_r, k_r = roc(r_h[hy == 1], r_h[hy == 0])
    out["ratio_curve"] = at_efficiency(e_r, k_r, grid)
    out["ratio_at"] = at_efficiency(e_r, k_r, marks)
    out["ratio_area"] = np.array([area(*roc(r_t[y == 1], r_t[y == 0])), area(e_r, k_r)])
    out["ratio_edges"], out["ratio"] = edges, ratio
    out["ratio_signal"], out["ratio_background"] = hs, hb
    return out


@functools.lru_cache(maxsize=1)
def run():
    """Everything measured (cached)."""
    e = T.events()
    y, hy = e["pool_y"], e["held_y"]
    return _cache("run", _build, (data, _build, T.fit_two, T.share, T.leaf_count),
                  pool=T.POOL, split=T.SPLIT, feature=FEATURE, rare=RARE,
                  ratio_bins=RATIO_BINS, sweep=RATIO_SWEEP, smooth=SMOOTH, grid=GRID,
                  n_train=len(y), n_held=len(hy))


def tree():
    """The Gini tree grown on the 8,000 training events, and its thresholds."""
    r = run()
    t = {k[len("tree_"):]: r[k] for k in r if k.startswith("tree_") and k != "tree_cuts"}
    return t, r["tree_cuts"]


CORE = (significance, kept, signal_side, scan, roc, area, at_efficiency, ratio_edges,
        ratio_cells, ratio_table, ratio_score, T.split_gain, T.impurity, T.events,
        T.two_features, T.grow, T.midpoints, T.binned, T.leaves_of)


# ====================================================================== the study
def _cut_record(r, f, tag=""):
    p = "%sscan%d_" % (tag, f)
    cuts, s, b = r[p + "cuts"], r[p + "s"], r[p + "b"]
    fb, gb = (int(v) for v in r[p + "best"])

    def one(k):
        return dict(cut=float(cuts[k]), s=float(s[k]), b=float(b[k]),
                    fom=float(r[p + "fom"][k]), gain=float(r[p + "gain"][k]))
    return dict(feature=f, name=T.FEATURES[f], below=bool(r[p + "below"]),
                by_fom=one(fb), by_gini=one(gb), same=fb == gb, apart=float(abs(
                    cuts[fb] - cuts[gb])))


def _held_at(cut, f, below):
    _, _, hx, hy = data()
    on = (hx[:, f] <= cut) == below
    return dict(efficiency=float(on[hy == 1].mean()), background=float(on[hy == 0].mean()))


def from_a_cut_to_a_tree():
    r = run()
    x, y, _, hy = data()
    S, B = int((y == 1).sum()), int((y == 0).sum())
    tr, cuts = tree()
    D = r["depths"]
    held_area, train_area = D[:, 3], D[:, 2]
    best = int(np.argmax(held_area))
    one = _cut_record(r, FEATURE)
    other = _cut_record(r, 1 - FEATURE)
    rare = _cut_record(r, FEATURE, "rare_")
    for rec in (one, other, rare):
        for k in ("by_fom", "by_gini"):
            rec[k].update(held=_held_at(rec[k]["cut"], rec["feature"], rec["below"]))
    f0, c0 = int(tr["feature"][0]), float(cuts[tr["feature"][0], tr["cut"][0]])
    grid = r["grid"]
    eff = float(r["stump_held"][0])
    show = sorted({1, 2, best + 1, len(D)})
    curves = np.array([r["cut_curve"]] + [r["depth_curves"][d - 1] for d in show]
                      + [r["ratio_curve"]])
    names = ["%s alone" % T.FEATURES[FEATURE]] + ["tree, depth %d" % d for d in show] + [
        "likelihood ratio"]
    # the plane: a few training events of each class, and the boxes at the depths shown
    pick = np.r_[np.arange(PLANE_EVENTS), S + np.arange(PLANE_EVENTS)]
    plane_depths = sorted({1, 2, 3, int(D[best, 0])})
    boxes = {"boxes_%d" % d: T.regions(tr, cuts, T.PLANE, max_depth=d) for d in plane_depths}
    edges = np.linspace(*T.PLANE[FEATURE], HIST_BINS + 1)
    hist = [np.histogram(x[y == k, FEATURE], edges)[0].astype(float) for k in (1, 0)]
    nodes = []
    for t in range(len(tr["feature"])):
        if tr["depth"][t] <= 2:
            f = int(tr["feature"][t])
            nodes.append(dict(node=t, depth=int(tr["depth"][t]), feature=f,
                              threshold=float(cuts[f, tr["cut"][t]]) if f >= 0 else None,
                              n=float(tr["n"][t]), n1=float(tr["n1"][t]),
                              left=int(tr["left"][t]), right=int(tr["right"][t])))
    p = "scan%d_" % FEATURE
    panels = (
        Panel("One feature, every threshold", "s / sqrt(s + b) of the training events kept "
              "below each threshold of the %s, and the Gini gain of the split there, each "
              "as a share of its largest" % T.FEATURES[FEATURE], "curve",
              dict(x=r[p + "cuts"], ys=np.array([r[p + "fom"] / r[p + "fom"].max(),
                                                 r[p + "gain"] / r[p + "gain"].max()]),
                   labels=["s / sqrt(s + b)", "Gini gain"], s=r[p + "s"], b=r[p + "b"],
                   fom=r[p + "fom"], gain=r[p + "gain"], edges=edges, signal=hist[0],
                   background=hist[1])),
        Panel("Cuts on cuts", "training events in the plane of the two features and the "
              "boxes of the Gini tree cut at depths %s" % ", ".join(map(str, plane_depths)),
              "points", dict(xy=x[pick], classes=y[pick], **boxes)),
        Panel("ROC on held-out events", "the share of held-out 9s kept against the share of "
              "held-out 4s kept: one feature cut at every threshold, the tree at several "
              "depths, the likelihood ratio of two 2-D histograms", "curve",
              dict(x=grid, ys=curves, labels=names)),
        Panel("How deep", "area under the ROC curve of the tree cut at each depth, on its "
              "training events and on held-out events", "curve",
              dict(x=D[:, 0], ys=np.array([train_area, held_area]),
                   labels=["training", "held out"], leaves=D[:, 1],
                   ratio=np.full(len(D), float(r["ratio_area"][1])))),
    )
    return Study(
        "From a cut to a tree", "An analysis cut is a tree of depth one; a tree is cuts on "
        "cuts, and it approaches the likelihood ratio", panels,
        dict(source="MNIST 4s (signal) and 9s (background), two measured features; a Gini "
                    "tree and 2-D histograms built here in numpy on the training pool, "
                    "every efficiency and area measured on the held-out events",
             dataset="MNIST", evidence=EVIDENCE, stand_in=T.STAND_IN,
             counterpart=T.COUNTERPART, split_seed=T.SPLIT, features=list(T.FEATURES),
             n_train=len(y), train_signal=S, train_background=B, n_held=len(hy),
             held_signal=int((hy == 1).sum()), held_background=int((hy == 0).sum()),
             figure_of_merit="s / sqrt(s + b) of the training events kept, as counted",
             feature=FEATURE, one_feature=one, other_feature=other,
             rare=dict(signal=RARE[0], background=RARE[1], **rare),
             root=dict(feature=f0, threshold=c0, gain=float(tr["gain"][0]),
                       left=[float(tr["n1"][tr["left"][0]]),
                             float(tr["n"][tr["left"][0]] - tr["n1"][tr["left"][0]])],
                       right=[float(tr["n1"][tr["right"][0]]),
                              float(tr["n"][tr["right"][0]] - tr["n1"][tr["right"][0]])]),
             stump_is_gini_cut=bool(f0 == FEATURE and np.isclose(
                 c0, one["by_gini"]["cut"])),
             top_nodes=nodes, nodes=int(len(tr["feature"])),
             leaves=int((tr["feature"] < 0).sum()), deepest=int(D[-1, 0]),
             depths=[dict(depth=int(row[0]), leaves=int(row[1]), train_area=float(row[2]),
                          held_area=float(row[3]),
                          background_at_stump=float(row[4]), background_at_80=float(row[5]))
                     for row in D],
             best_depth=int(D[best, 0]), best_area=float(held_area[best]),
             shown_depths=show, plane_depths=plane_depths,
             cut_area=float(r["cut_area"]),
             stump_held=dict(efficiency=eff, background=float(r["stump_held"][1])),
             cut_background_at_stump=float(r["cut_at"][0]),
             cut_background_at_80=float(r["cut_at"][1]),
             ratio=dict(bins=RATIO_BINS, smooth=SMOOTH, train_area=float(r["ratio_area"][0]),
                        held_area=float(r["ratio_area"][1]),
                        background_at_stump=float(r["ratio_at"][0]),
                        background_at_80=float(r["ratio_at"][1]),
                        sweep=dict(bins=list(RATIO_SWEEP), held_area=_floats(
                            r["ratio_sweep"]))),
             limitations=["one tree, grown on one division of the events; the ratio's "
                          "histograms have %d equal-count bins a feature, fixed in advance "
                          "(%s bins give held-out areas from %.3f to %.3f)"
                          % (RATIO_BINS, "–".join(map(str, RATIO_SWEEP[::len(RATIO_SWEEP)
                                                                       - 1])),
                             min(r["ratio_sweep"]), max(r["ratio_sweep"])),
                          "the likelihood ratio is the best test only for the true "
                          "densities; this one is estimated, so a tree could in principle "
                          "beat it on held-out events",
                          "s / sqrt(s + b) is taken on the events as counted, an even mix "
                          "of 4s and 9s; with signal rare it chooses a tighter cut"]))


STUDIES = {NAME: from_a_cut_to_a_tree}

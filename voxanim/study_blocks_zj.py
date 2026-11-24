"""The numbers behind animation 279 of "Blocks and code" (group ``zj``, "Jets as sets of
particles"): an MLP reading a jet's particles as a list, against a Deep Sets network, and what
shuffling the list does to each. Everything comes from :mod:`voxanim.study_jets`, the numpy
twin of ``examples/jets_as_sets.py`` (checked against torch by
``tests/test_jets_equivalence.py``). Made-up jets: 6 to 20 particles, z from a Dirichlet(0.7),
one-prong (one core, spread 0.1 to 0.35) or two-prong (two cores 0.25 to 1.0 apart, each
spread 0.1); 40,000 to train, 4,000 held out; one run of each model.

* ``jets-as-sets``: two held-out jets drawn in (Δη, Δφ), a dot per particle sized by z (a
  two-prong one followed all through, :func:`followed`, and a one-prong one beside it); the
  followed jet as its list, sorted by z, padded to 20 rows, with its mask; the MLP reading the
  60 numbers in a row, Deep Sets running phi on each row, adding the real rows, then rho; both
  models' answers for the followed jet and their held-out accuracy; the jet's rows shuffled
  (the script's shuffle, seed 3), the answers again; over the 4,000 held-out jets, shuffled:
  the accuracy, the answers that flip and how far the answers move; the rows sorted again.

The followed jet is chosen by a rule, said on screen: a two-prong jet that both models get
right in sorted order and whose MLP answer flips when shuffled (as 754 of the 4,000 do), with
11 to 14 particles, all inside |Δη|, |Δφ| < 0.75, the one whose move is nearest the median
move of the jets that flip. Most jets move far less (the median over all is 0.035): the
screen says so. The one-prong jet beside it: one that both get right (and whose MLP answer does
not flip) with as many particles, of median width among those.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_jets as J
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("made-up jets: examples/jets_as_sets.py (6-20 particles; one or two prongs); the "
          "MLP and Deep Sets trained once each in numpy, as the script trains them (Adam, "
          "6,000 steps of 64 jets on 40,000); measured on 4,000 held-out jets, shuffled once "
          "(seed 3)")
REACH = 0.75                 # the jets drawn: every particle inside |Δη|, |Δφ| < REACH
SPAN = 0.8                   # the (Δη, Δφ) panels show -SPAN..SPAN
LOSS_EVERY = 50              # the loss curves: means over 50 steps


def pct(v):
    """An accuracy as the screen writes it: ``90.2%``."""
    return "%.1f%%" % (100 * float(v))


def p_text(v):
    """A probability as the screen writes it: ``0.98``."""
    return "%.2f" % float(v)


def tiny(v):
    """A very small number in words a student can read: ``0.0000004``."""
    v = float(v)
    if v == 0:
        return "0"
    digits = int(np.ceil(-np.log10(v)))
    return ("%.*f" % (digits, v)).rstrip("0")


@functools.lru_cache(maxsize=1)
def followed():
    """``(two-prong jet, one-prong jet)`` indices into the held-out set, by the rule in the
    module docstring."""
    X, M, y, _Xs = J.test_set()
    p = J.probs()
    n = M.sum(1)
    inside = np.abs(X[:, :, 1:]).max(axis=(1, 2)) < REACH
    right = ((p["mlp"] > 0.5) == y) & ((p["ds"] > 0.5) == y)
    flip = (p["mlp"] > 0.5) != (p["mlp_shuffled"] > 0.5)
    moved = np.abs(p["mlp"] - p["mlp_shuffled"])
    middle = float(np.median(moved[flip]))
    two = np.where(inside & right & flip & (y == 1) & (n >= 11) & (n <= 14))[0]
    w = int(two[np.argmin(np.abs(moved[two] - middle))])
    one = np.where(inside & right & ~flip & (y == 0) & (n == n[w]) & (p["mlp"] < 0.2)
                   & (p["ds"] < 0.2))[0]
    # of those, the one of median width (z-weighted distance from the axis): a typical core
    width = np.array([float(X[i, :, 0] @ np.hypot(X[i, :, 1], X[i, :, 2])) for i in one])
    return w, int(one[np.argsort(width)[len(one) // 2]])


def phi_rows(x, mask):
    """Deep Sets' phi on each row of one jet (trained weights), the pads' rows set to 0 by the
    mask, and their sum: ``(rows [20, 32], sum [32])``."""
    phi = J.weights("phi")
    h = np.maximum(J._two(phi, x[None])[0][0], 0) * mask[:, None]
    return h, h.sum(0)


def jet_info(i):
    X, M, y, Xs = J.test_set()
    p = J.probs()
    n = int(M[i].sum())
    # the shuffle as moves: the row now at slot s was sorted row perm[s]
    perm = [int(np.where((X[i, :n] == Xs[i, s]).all(1))[0][0]) for s in range(n)]
    return dict(index=i, n=n, pads=J.MAX - n, two=bool(y[i]), rows=X[i].tolist(),
                mask=M[i].tolist(), shuffled=Xs[i].tolist(), perm=perm,
                p_mlp=float(p["mlp"][i]), p_mlp_shuffled=float(p["mlp_shuffled"][i]),
                p_mlp_again=float(p["mlp_again"][i]), p_ds=float(p["ds"][i]),
                p_ds_shuffled=float(p["ds_shuffled"][i]),
                ds_moved=float(abs(p["ds"][i] - p["ds_shuffled"][i])))


# ====================================================================== 279
def jets():
    r = J.results()
    w, o = followed()
    W, Q = jet_info(w), jet_info(o)
    X, M, _y, Xs = J.test_set()
    rows, total = phi_rows(X[w], M[w])
    rows_s, total_s = phi_rows(Xs[w], M[w])
    p = J.probs()
    moved = np.abs(p["mlp"] - p["mlp_shuffled"])
    flip = (p["mlp"] > 0.5) != (p["mlp_shuffled"] > 0.5)
    mlp_w = J.weights("mlp")["1.weight"]
    loss = {m: J.run()["loss_" + m].reshape(-1, LOSS_EVERY).mean(1) for m in J.MODELS}
    right, cnt = r["right"], r["count"]
    m = dict(
        source=SOURCE, max_rows=J.MAX, features=J.FEATURES, n_train=J.N_TRAIN,
        n_test=J.N_TEST, steps=J.STEPS, batch=J.BATCH, lr=J.LR, span=SPAN,
        followed=W, other=Q,
        phi_rows=rows.tolist(), phi_sum=total.tolist(), phi_sum_shuffled=total_s.tolist(),
        phi_pad=np.maximum(J._two(J.weights("phi"), np.zeros((1, J.FEATURES), np.float32))[0],
                           0)[0].tolist(),
        phi_sum_moved=float(np.abs(total - total_s).max()),
        phi_rows_shuffled=rows_s.tolist(),
        mlp_first=mlp_w[:12, :12].tolist(),
        right=right, right_text={k: pct(v) for k, v in right.items()},
        train_right=r["train_right"],
        p_text=dict(mlp=p_text(W["p_mlp"]), mlp_shuffled=p_text(W["p_mlp_shuffled"]),
                    mlp_again=p_text(W["p_mlp_again"]), ds=p_text(W["p_ds"]),
                    ds_shuffled=p_text(W["p_ds_shuffled"]), other_mlp=p_text(Q["p_mlp"]),
                    other_ds=p_text(Q["p_ds"])),
        changed=r["mlp"]["changed"], changed_text=commas(r["mlp"]["changed"]),
        changed_ds=r["ds"]["changed"],
        changed_share=r["mlp"]["changed"] / J.N_TEST,
        mean_moved=r["mlp"]["mean_moved"], median_moved=r["mlp"]["median_moved"],
        median_moved_text="%.3f" % r["mlp"]["median_moved"],
        mean_moved_text="%.2f" % r["mlp"]["mean_moved"],
        flipped_median_moved=float(np.median(moved[flip])),
        over_10=r["mlp"]["over_10"],
        ds_max_moved=r["ds"]["max_moved"], ds_max_moved_text=tiny(r["ds"]["max_moved"]),
        ds_bound_text=tiny(10.0 ** np.ceil(np.log10(r["ds"]["max_moved"]))),
        ds_bitwise_same=r["ds"]["bitwise_same"],
        sorted_again_same=r["sorted_again_same"], mlp_again_same=r["mlp_again_same"],
        count=cnt, count_text={k: commas(v) for k, v in cnt.items()},
        loss=r["loss"],
        code_jet=list(code_lines("js-jet")), code_pad=list(code_lines("js-pad")),
        code_mlp=list(code_lines("js-mlp")), code_sum=list(code_lines("js-sum")),
        code_shuffle=list(code_lines("js-shuffle")), code_measure=list(code_lines("js-measure")),
        code_again=list(code_lines("js-again")),
        limitations=[
            "made-up jets, not simulated ones: the recipe (Dirichlet(0.7) momentum fractions, "
            "one Gaussian core or two) is the script's; real jets need a detector simulation",
            "one run of each model (one seed of starting weights): their accuracies are that "
            "run's; the script, run in torch from its own seed, gives similar ones",
            "the followed jet is chosen by a rule (two prongs, both models right, its MLP "
            "answer flips when shuffled, its move the median of those that flip); most jets "
            "move far less: the median move over all 4,000 is %.3f" % r["mlp"]["median_moved"],
            "Deep Sets' answer cannot depend on the order in exact arithmetic; in float32 the "
            "sum rounds differently, moving answers by at most %s (%d of 4,000 bit for bit "
            "the same) and flipping none" % (tiny(r["ds"]["max_moved"]),
                                              r["ds"]["bitwise_same"]),
            "the shuffle keeps the pads last; each jet is shuffled once (seed 3)"])
    xs = np.arange(len(loss["mlp"]), dtype=float) * LOSS_EVERY
    panels = (
        Panel("Training loss", "mean over each 50 steps; one run each", "curve",
              dict(x=xs, ys=np.array([loss["mlp"], loss["ds"]]), labels=["MLP", "Deep Sets"])),
        Panel("Held-out accuracy, 4,000 jets", "sorted, shuffled, sorted again", "bars",
              dict(values=np.array([right["mlp"], right["mlp_shuffled"], right["mlp_again"],
                                    right["ds"], right["ds_shuffled"]]),
                   labels=["MLP sorted", "MLP shuffled", "MLP sorted again", "Deep Sets sorted",
                           "Deep Sets shuffled"])),
        Panel("The followed jet", "Δη against Δφ of its %d particles" % W["n"], "points",
              dict(xy=np.array(W["rows"])[:W["n"], 1:])),
        Panel("How far each answer moved when shuffled", "sorted moves, 4,000 jets", "curve",
              dict(x=np.linspace(0, 1, J.N_TEST),
                   ys=np.array([np.sort(moved), np.sort(np.abs(p["ds"] - p["ds_shuffled"]))]),
                   labels=["MLP", "Deep Sets"])),
    )
    return Study("Jets as sets of particles",
                 "Shuffled, the MLP gets %s right (%s sorted); Deep Sets %s either way"
                 % (pct(right["mlp_shuffled"]), pct(right["mlp"]), pct(right["ds"])), panels, m)


STUDIES = {"jets-as-sets": jets}

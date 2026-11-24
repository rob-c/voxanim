"""The numbers behind animations 245-248 of "Blocks and code" (group ``v``, the first four of
"Recognising faces"): a face as numbers, the encoder that turns it into 32, the triplet loss
that trains it, and verification. Everything comes from :mod:`voxanim.study_faces`, the
numpy twin of ``examples/faces.py``, trained once; the photos are the Olivetti faces
(:mod:`voxanim.olivetti`). People are only ever named by their number.

* ``faces-a-face-is-numbers``: a photo is 64 x 64 = 4,096 numbers from 0 to 1 (photo 300,
  the first photo of person 30, the first unseen person); the 400 photos, 40 people x 10,
  people 0-29 for training and 30-39 never trained on. The distance of two photos pixel by
  pixel (each photo's 4,096 numbers scaled to length 1, as ``recognition("pixels")``
  compares them): person 30's first two photos are 0.340 apart, person 30's and person 31's
  first photos 0.336 (closer, though two people: picked by that rule, the first photos,
  not searched for). Over the 4,950 pairs of people 30-39, one person's pairs lie between
  0.06 and 0.42, two people's between 0.17 and 0.52: they overlap.
* ``faces-an-embedding``: the encoder of region "fc-encoder" on photo 300 at 32 x 32:
  [1, 32, 32] -> [8, 16, 16] -> [16, 8, 8] -> [32, 4, 4] -> 512 -> 32 numbers, of length
  0.09, scaled to length 1 (``F.normalize``); 22,304 numbers learned (80 + 1,168 + 4,640 +
  16,416).
* ``faces-triplets``: the first triplet of the first step (``triplets(0)``): photo 144 as
  the anchor, 149 (the same person, 14) as the positive, 30 (person 3) as the negative.
  Before training the anchor is 0.04 from the positive and 0.09 from the negative: the
  loss is 0.04 - 0.09 + 0.5 = 0.45. After 1,500 steps, 0.24 and 1.27: the negative is
  more than 0.5 further, the loss 0. Over training (``rec_run()``), the loss per step and
  the share of the 32 triplets not yet 0.5 apart, each a mean of 25 steps.
* ``faces-verification``: the 4,950 pairs of the 100 photos of people 30-39 (450 of one
  person, 4,500 of two), their distances through the trained encoder, the threshold 0.72
  chosen on the training people's pairs (never on these): 86 false accepts (1.9% of 4,500)
  and 111 false rejects (24.7% of 450), 4,753 right. The equal error rate on these pairs is
  12.4% against 22.0% for raw pixels (18.0% for the untrained network); at their own
  thresholds chosen on people 0-29, pixels make 390 mistakes, the encoder 197.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_faces as F
from .blocks import code_lines, commas, shape
from .study import Panel, Study

PHOTO = 300                     # person 30's first photo: the first unseen photo
PATCH = (24, 20, 3)             # 245: the 3 x 3 tiles shown with their numbers (row, col, side)
SAME_PAIR = (300, 301)          # 245: person 30's first two photos
DIFF_PAIR = (300, 310)          # 245: person 30's and person 31's first photos
THUMB = 16                      # a thumbnail: the photo averaged down to 16 x 16
PIXEL_BINS = np.round(np.arange(0.0, 0.6001, 0.025), 3)   # 245: the pixel distances' bins
BINS = np.round(np.arange(0.0, 2.0001, 0.05), 3)           # 248: the encoder's distances' bins
TRIPLET = (0, 0)                # 247: step 0's first triplet
MEAN_OF = 25                    # 247: each point of the curves is the mean of 25 steps
LAST = 100                      # 247: "at the end": the mean of the last 100 steps
SOURCE = ("the Olivetti faces (AT&T Laboratories Cambridge, 1992-94): 40 people, 10 photos "
          "each; people 0-29 train, people 30-39 are only tested on; examples/faces.py "
          "trained once (voxanim.study_faces, its numpy twin)")
PEOPLE_NOTE = ("people are named only by their number; nothing is said or guessed about who "
               "they are")


def pct(share, places=1):
    """``12.4%``."""
    return "%.*f%%" % (places, 100 * share)


def dist(d, places=2):
    """A distance as shown: ``0.34``."""
    return "%.*f" % (places, float(d))


@functools.lru_cache(maxsize=1)
def thumbnails():
    """Every photo averaged down to :data:`THUMB` x :data:`THUMB` (``F.interpolate`` "area"):
    ``[400, 16, 16]``."""
    x = F.photos()[0]
    return np.stack([F.resize(img, THUMB) for img in x])


@functools.lru_cache(maxsize=1)
def pixel_vectors():
    """Each photo's 4,096 numbers scaled to length 1: what ``recognition("pixels")``
    compares."""
    return F.unit(F.photos()[0].reshape(400, -1))


def pixel_distance(a, b):
    v = pixel_vectors()
    return float(np.linalg.norm(v[a] - v[b]))


def histogram(d, bins):
    """How many of ``d`` fall in each bin (the last bin closed)."""
    return np.histogram(np.clip(d, bins[0], bins[-1]), bins)[0]


# ====================================================================== 245
def a_face_is_numbers():
    x, who = F.photos()
    img = x[PHOTO]
    r0, c0, side = PATCH
    patch = img[r0:r0 + side, c0:c0 + side]
    px = F.recognition("pixels")
    d_same, d_diff = np.asarray(px["d_same"], float), np.asarray(px["d_diff"], float)
    same, diff = pixel_distance(*SAME_PAIR), pixel_distance(*DIFF_PAIR)
    if not (who[SAME_PAIR[0]] == who[SAME_PAIR[1]] and who[DIFF_PAIR[0]] != who[DIFF_PAIR[1]]):
        raise ValueError("the example pairs are not one person and two people")
    lo, hi = float(d_diff.min()), float(d_same.max())     # where both kinds of pair are
    in_same = int(((d_same >= lo) & (d_same <= hi)).sum())
    in_diff = int(((d_diff >= lo) & (d_diff <= hi)).sum())
    h_same, h_diff = histogram(d_same, PIXEL_BINS), histogram(d_diff, PIXEL_BINS)
    n = int(img.size)
    meta = dict(
        source=SOURCE, evidence="measured on Olivetti faces",
        photo=PHOTO, photo_person=int(who[PHOTO]), image=img, side=int(img.shape[0]),
        numbers=n, numbers_text=commas(n), photos=len(x), people=int(who.max()) + 1,
        per_person=int((who == 0).sum()), train_people=F.TRAIN_PEOPLE,
        photos_shape=shape(x.shape),
        patch=patch, patch_at=(r0, c0), patch_side=side,
        patch_text=[["%.2f" % v for v in row] for row in patch],
        lo_value=float(x.min()), hi_value=float(x.max()),
        thumbs=thumbnails(), who=who,
        same_pair=SAME_PAIR, diff_pair=DIFF_PAIR,
        same_people=[int(who[i]) for i in SAME_PAIR], diff_people=[int(who[i]) for i in DIFF_PAIR],
        same_images=[x[i] for i in SAME_PAIR], diff_images=[x[i] for i in DIFF_PAIR],
        same_distance=same, diff_distance=diff,
        same_text=dist(same, 3), diff_text=dist(diff, 3),
        pairs=int(px["pairs"]), pairs_text=commas(px["pairs"]),
        same_pairs=len(d_same), diff_pairs=len(d_diff),
        same_pairs_text=commas(len(d_same)), diff_pairs_text=commas(len(d_diff)),
        same_range=(float(d_same.min()), hi), diff_range=(lo, float(d_diff.max())),
        same_range_text=(dist(d_same.min()), dist(hi)),
        diff_range_text=(dist(lo), dist(d_diff.max())),
        overlap=(lo, hi), overlap_text=(dist(lo), dist(hi)),
        in_overlap_same=in_same, in_overlap_diff=in_diff,
        in_overlap_same_text=commas(in_same), in_overlap_diff_text=commas(in_diff),
        bins=PIXEL_BINS, hist_same=h_same, hist_diff=h_diff,
        share_same=h_same / len(d_same), share_diff=h_diff / len(d_diff),
        code_data=list(code_lines("fc-data")),
        limitations=[
            "distance: each photo's 4,096 numbers scaled to length 1, then the length of "
            "the difference (as recognition('pixels') compares them); other pixel "
            "distances give other numbers",
            "the example pairs are picked by a rule (person 30's first two photos; person "
            "30's and person 31's first photos), not searched for; that the two-people pair "
            "is the closer is what the rule found",
            "the 4,950 pairs are those of the ten unseen people, 30-39",
            PEOPLE_NOTE])
    panels = (
        Panel("A photo is numbers", "photo %d (person %d): 64 x 64 numbers from 0 to 1"
              % (PHOTO, who[PHOTO]), "images", dict(values=img[None], labels=["photo %d"
                                                                              % PHOTO])),
        Panel("40 people, 10 photos each", "every photo averaged down to 16 x 16; person = "
              "photo // 10", "images", dict(values=thumbnails()[::10],
                                            labels=["person %d" % p for p in range(40)])),
        Panel("Two pairs", "pixel distances: person 30's first two photos, and person 30's "
              "and 31's first photos", "bars",
              dict(values=np.array([same, diff]), labels=["one person", "two people"])),
        Panel("All pairs of people 30-39", "pixel distances of the 450 pairs of one person "
              "and the 4,500 of two people, as shares", "curve",
              dict(x=(PIXEL_BINS[:-1] + PIXEL_BINS[1:]) / 2,
                   ys=np.array([h_same / len(d_same), h_diff / len(d_diff)]),
                   labels=["one person", "two people"])),
    )
    return Study("A face is 4,096 numbers", "Pixel distances of one person and of two people "
                 "overlap", panels, meta)


# ====================================================================== 246
def layer_counts():
    """``(name, weights, biases)`` of each layer that learns, in the encoder's order."""
    P = F.rec_init()
    names = ("block(1, 8)", "block(8, 16)", "block(16, 32)", "Linear(512, 32)")
    keys = ("0", "3", "6", "10")
    return [(nm, int(P[k + ".weight"].size), int(P[k + ".bias"].size))
            for nm, k in zip(names, keys, strict=True)]


def busiest(maps):
    """The map with the widest spread of values (the front card shows it)."""
    spread = np.ptp(maps.reshape(len(maps), -1), axis=1)
    return int(spread.argmax())


def an_embedding():
    P = F.rec_params()
    x = F.small()[PHOTO:PHOTO + 1]
    e, (flat, hshape, cache) = F.rec_forward(P, x, keep_cache=True)
    outs = [cache[1][0][0], cache[2][0][0], flat.reshape(hshape)[0]]
    shown = [busiest(m) for m in outs]
    v = F.unit(e)[0]
    e = e[0]
    counts = layer_counts()
    total = sum(w + b for _n, w, b in counts)
    if total != F.count(P):
        raise ValueError("the layers' counts do not add up")
    shapes = [(1, 32, 32)] + [m.shape for m in outs]
    who = F.photos()[1]
    meta = dict(
        source=SOURCE, evidence="measured on Olivetti faces",
        photo=PHOTO, person=int(who[PHOTO]), image=x[0, 0], full=F.photos()[0][PHOTO],
        shapes=[shape(s) for s in shapes], sizes=[int(np.prod(s)) for s in shapes],
        sizes_text=[commas(np.prod(s)) for s in shapes],
        maps=[m[i] for m, i in zip(outs, shown, strict=True)], map_index=shown,
        channels=[int(s[0]) for s in shapes],
        flat=flat[0], flat_len=int(flat.shape[1]), flat_text=commas(flat.shape[1]),
        e=e, v=v, dims=int(len(v)),
        length=float(np.linalg.norm(e)), length_text=dist(np.linalg.norm(e)),
        unit_length=float(np.linalg.norm(v)), unit_length_text=dist(np.linalg.norm(v)),
        v_lo=float(v.min()), v_hi=float(v.max()),
        counts=[dict(name=nm, weights=w, biases=b, total=w + b, text=commas(w + b))
                for nm, w, b in counts],
        total=total, total_text=commas(total),
        code_encoder=list(code_lines("fc-encoder")), code_block=list(code_lines("fc-block")),
        code_normalize=code_lines("fc-triplet")[0],
        limitations=[
            "photo %d (person %d's first photo, never trained on), averaged to 32 x 32 as "
            "the script's small does" % (PHOTO, who[PHOTO]),
            "each block's front card shows the one of its maps with the widest spread of "
            "values (maps %s); the others are cards behind it" % shown,
            "the 32 numbers are the trained encoder's (1,500 steps); they mean nothing by "
            "themselves, only distances between them do",
            PEOPLE_NOTE])
    panels = (
        Panel("One photo through the blocks", "photo %d at 32 x 32, then one map of each "
              "block's output" % PHOTO, "images",
              dict(values=np.stack([F.resize(m, 32) if m.shape[0] != 32 else m
                                    for m in [x[0, 0]] + meta["maps"]]),
                   labels=meta["shapes"][:4])),
        Panel("Numbers at each stage", "how many numbers the photo is at each stage", "bars",
              dict(values=np.array(meta["sizes"] + [meta["flat_len"], meta["dims"]], float),
                   labels=meta["shapes"] + ["flat", "out"])),
        Panel("32 numbers, then length 1", "the encoder's 32 numbers for photo %d, before "
              "and after scaling to length 1" % PHOTO, "matrix",
              dict(values=np.array([e, v]))),
        Panel("Numbers learned", "weights and biases of each layer: %s in all"
              % commas(total), "bars",
              dict(values=np.array([w + b for _n, w, b in counts], float),
                   labels=[nm for nm, _w, _b in counts])),
    )
    return Study("A face as 32 numbers", "Three blocks and a linear layer turn a 32 x 32 face "
                 "into 32 numbers of length 1; 22,304 numbers learned", panels, meta)


# ====================================================================== 247
def triplet_distances(step):
    """The example triplet's distances and loss with the weights of ``step``."""
    i, j, k = F.triplets(TRIPLET[0])
    t = TRIPLET[1]
    loss, _G, d = F.triplet_loss_and_grads(F.rec_params(step), i[t:t + 1], j[t:t + 1],
                                           k[t:t + 1])
    ap, an = float(d["ap"][0]), float(d["an"][0])
    return dict(ap=ap, an=an, loss=loss, ap_text=dist(ap), an_text=dist(an),
                loss_text=dist(loss), short=bool(ap - an + F.MARGIN > 0))


def means(a, n=MEAN_OF):
    a = np.asarray(a, float)
    return a[:len(a) // n * n].reshape(-1, n).mean(1)


def triplets_study():
    i, j, k = F.triplets(TRIPLET[0])
    t = TRIPLET[1]
    ids = [int(i[t]), int(j[t]), int(k[t])]
    who = F.photos()[1]
    people = [int(who[q]) for q in ids]
    if not (people[0] == people[1] != people[2] and max(people) < F.TRAIN_PEOPLE):
        raise ValueError("not an anchor, a positive and a negative of the training people")
    before, after = triplet_distances(0), triplet_distances(F.REC_STEPS)
    run = F.rec_run()
    loss, active = np.asarray(run["loss"], float), np.asarray(run["active"], float)
    lm, am = means(loss), means(active)
    steps = np.arange(1, len(lm) + 1) * MEAN_OF        # each mean ends at this step
    meta = dict(
        source=SOURCE, evidence="measured on Olivetti faces",
        step=TRIPLET[0], index=t, ids=ids, people=people,
        images=[F.small()[q, 0] for q in ids], full=[F.photos()[0][q] for q in ids],
        margin=F.MARGIN, margin_text=dist(F.MARGIN, 1), batch=F.TRIPLETS,
        before=before, after=after,
        steps=len(loss), steps_text=commas(len(loss)), mean_of=MEAN_OF,
        curve_steps=steps.astype(float), loss_means=lm, active_means=am,
        first_loss=float(loss[0]), first_loss_text=dist(loss[0]),
        first_active=float(active[0]), first_active_text=pct(active[0], 0),
        first_short=int(round(active[0] * F.TRIPLETS)),
        last_loss=float(loss[-LAST:].mean()), last_loss_text="%.3f" % loss[-LAST:].mean(),
        last_active=float(active[-LAST:].mean()), last_active_text=pct(active[-LAST:].mean()),
        last=LAST, loss_hi=float(lm.max()),
        code_triplet=list(code_lines("fc-triplet")),
        limitations=[
            "the triplet shown is the first of the first step's 32 (triplets(0)): photos "
            "%d and %d of person %d, photo %d of person %d, all training people"
            % (ids[0], ids[1], people[0], ids[2], people[2]),
            "distances as F.triplet_margin_loss computes them (with eps 1e-6), between "
            "vectors of length 1",
            "each point of the curves is the mean of %d steps; a step's share counts the "
            "32 triplets whose negative is not yet 0.5 further than the positive" % MEAN_OF,
            "one training run; the loss is on the training triplets, not on unseen people",
            PEOPLE_NOTE])
    panels = (
        Panel("Anchor, positive, negative", "the first triplet of step 0: people %d, %d "
              "and %d" % tuple(people), "images",
              dict(values=np.stack(meta["images"]),
                   labels=["anchor", "positive", "negative"])),
        Panel("Distances before and after", "anchor to positive and to negative, before "
              "training and after 1,500 steps", "bars",
              dict(values=np.array([before["ap"], before["an"], after["ap"], after["an"]]),
                   labels=["before: positive", "before: negative", "after: positive",
                           "after: negative"])),
        Panel("The loss", "the triplet loss per step, each point a mean of %d steps"
              % MEAN_OF, "curve", dict(x=steps.astype(float), ys=lm[None], labels=["loss"])),
        Panel("Triplets not yet 0.5 apart", "the share of each step's 32 triplets still in "
              "the loss, each point a mean of %d steps" % MEAN_OF, "curve",
              dict(x=steps.astype(float), ys=am[None], labels=["share"])),
    )
    return Study("Triplets: same closer than different", "The triplet loss pushes another "
                 "person 0.5 further than another photo of the same person", panels, meta)


# ====================================================================== 248
def verification():
    r, px, un = (F.recognition(n) for n in ("trained", "pixels", "untrained"))
    d_same, d_diff = np.asarray(r["d_same"], float), np.asarray(r["d_diff"], float)
    t = float(r["threshold"])
    if abs(t - F.trained_threshold()) > 1e-6:
        raise ValueError("the threshold is not the one chosen on the training people")
    fa, fr = int((d_diff < t).sum()), int((d_same >= t).sum())
    if (fa, fr) != (r["false_accept"], r["false_reject"]):
        raise ValueError("false accepts and rejects do not match the study's")
    h_same, h_diff = histogram(d_same, BINS), histogram(d_diff, BINS)
    who = F.photos()[1]
    test = np.nonzero(who >= F.TRAIN_PEOPLE)[0]
    mistakes = dict(trained=fa + fr, pixels=int(px["false_accept"] + px["false_reject"]),
                    untrained=int(un["false_accept"] + un["false_reject"]))
    eer = dict(trained=float(r["test_eer"]), pixels=float(px["test_eer"]),
               untrained=float(un["test_eer"]))
    meta = dict(
        source=SOURCE, evidence="measured on Olivetti faces",
        photos=len(test), thumbs=thumbnails()[test], test_people=(int(who[test].min()),
                                                                  int(who[test].max())),
        pairs=int(r["pairs"]), pairs_text=commas(r["pairs"]),
        same_pairs=len(d_same), diff_pairs=len(d_diff),
        same_pairs_text=commas(len(d_same)), diff_pairs_text=commas(len(d_diff)),
        bins=BINS, hist_same=h_same, hist_diff=h_diff,
        share_same=h_same / len(d_same), share_diff=h_diff / len(d_diff),
        same_range=(float(d_same.min()), float(d_same.max())),
        diff_range=(float(d_diff.min()), float(d_diff.max())),
        threshold=t, threshold_text=dist(t), train_people=F.TRAIN_PEOPLE,
        false_accept=fa, false_reject=fr, right=int(r["right"]),
        right_text=commas(r["right"]),
        far=fa / len(d_diff), frr=fr / len(d_same),
        far_text=pct(fa / len(d_diff)), frr_text=pct(fr / len(d_same)),
        eer=eer, eer_text={k: pct(v) for k, v in eer.items()},
        mistakes=mistakes, mistakes_text={k: commas(v) for k, v in mistakes.items()},
        pixel_threshold=float(px["threshold"]),
        pixel_false_accept=int(px["false_accept"]), pixel_false_reject=int(px["false_reject"]),
        code_verify=list(code_lines("fc-verify")),
        printed="right: tensor(%d)" % int(r["right"]),
        limitations=[
            "the threshold (%.2f) is chosen on the training people's pairs, where false "
            "accepts and false rejects are equally common; on the unseen people it gives "
            "%s false accepts and %s false rejects: it does not balance them there"
            % (t, pct(fa / len(d_diff)), pct(fr / len(d_same))),
            "the equal error rate is measured on these 4,950 pairs themselves, at the "
            "threshold where both mistakes are equally common: a way to compare the three, "
            "not a threshold anyone could set in advance",
            "raw pixels: each photo's 4,096 numbers scaled to length 1; the untrained "
            "network: the same encoder before its first step",
            "ten unseen people, 100 photos, one training run: a small test",
            PEOPLE_NOTE])
    order = ("pixels", "untrained", "trained")
    panels = (
        Panel("One person", "distances of the 450 pairs of one person, as shares", "bars",
              dict(values=h_same / len(d_same), labels=["%.2f" % b for b in BINS[:-1]])),
        Panel("Two people", "distances of the 4,500 pairs of two people, as shares", "bars",
              dict(values=h_diff / len(d_diff), labels=["%.2f" % b for b in BINS[:-1]])),
        Panel("Mistakes at %.2f" % t, "false accepts and false rejects at the threshold "
              "chosen on people 0-29", "bars",
              dict(values=np.array([fa, fr], float),
                   labels=["false accepts", "false rejects"])),
        Panel("Equal error rate", "where both mistakes are equally common, on the 4,950 "
              "pairs: pixels, the untrained network, the trained encoder", "bars",
              dict(values=np.array([eer[k] for k in order]), labels=list(order))),
    )
    return Study("Same person or not?", "At a threshold chosen on the training people, "
                 "4,753 of 4,950 pairs right; equal error rate 12.4% against 22.0% for "
                 "pixels", panels, meta)


STUDIES = {"faces-a-face-is-numbers": a_face_is_numbers, "faces-an-embedding": an_embedding,
           "faces-triplets": triplets_study, "faces-verification": verification}

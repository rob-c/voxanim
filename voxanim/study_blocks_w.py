"""The numbers behind animations 249-251 of "Blocks and code" (group ``w``, the last three of
"Recognising faces"): identification, the two models together, and what 40 people cannot
show. Everything comes from :mod:`voxanim.study_faces`, the numpy twin of
``examples/faces.py`` (one detector and one recogniser, each trained once).

* ``faces-identification``: one photo each of people 30-34 is enrolled (region
  "fc-identify": ``gallery = v[:50:10]``, photos 300, 310, ..., 340); each of their other 45
  photos is a probe, named after the nearest enrolled face. Two probes are followed, each
  picked by a rule (:func:`followed`): the first whose nearest enrolled face is nearer
  than the threshold (photo 305, person 30, 0.18 from person 30's enrolled photo) and the
  first whose nearest is not (photo 301, also person 30: nearest person 31 at 0.81, so
  "unsure", and the name would have been wrong). Over the 45: the nearest is right for 36,
  exactly as many as with raw pixels (36): the encoder does not help this identification
  (it helps verification: equal error rate 12.4% against 22.0%). 14 of the 45 are unsure.
  The 50 photos of people 35-39, never enrolled, are strangers: 6 are nearer than the
  threshold to an enrolled face and would be let in (region "fc-strangers").
* ``faces-the-whole-pipeline``: :func:`~voxanim.study_faces.pipeline_scene`, the first
  seeded scene of the enrolled people with three faces (picked by that rule, not by how
  well it goes; never their enrolled photo): the detector's three boxes, each face cut out
  and resized to 32 x 32, its 32 numbers, the nearest enrolled person and the distance
  (region "fc-pipeline"). All three are named right; person 34's at 0.84, above the
  threshold, so "unsure". Then :func:`~voxanim.study_faces.pipeline_rates` over 200 such
  scenes: 341 of 391 faces found; of those 341, 227 named right (67%) from the detector's
  box; from the true box, 320 of the 391 (82%). Boxes a little off (IoU 0.76 on average)
  change the crop and cost names.
* ``faces-what-40-people-cannot-show``: the false-reject and false-accept rates of each of
  the ten unseen people (``recognition()["per_person_frr"]`` / ``["per_person_far"]``),
  by person number only: from 0 to 73% of a person's 45 same-person pairs rejected. Then
  what this data cannot measure (said in words, about the data, not about the people):
  40 people, ten photos each, one lab, 1992-94.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_faces as F
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("the Olivetti faces (AT&T Laboratories Cambridge, 1992-94: ten photos of each of 40 "
          "people); people 0-29 trained the models, people 30-39 are only tested on; one "
          "detector and one recogniser trained once as examples/faces.py trains them "
          "(voxanim.study_faces, its numpy twin)")
TAG = "measured on Olivetti faces: "       # the evidence tag of a measured phase
ABOUT = "about the data: "                 # ... and of a phase about the data itself


def measured(what):
    """The evidence tag of a phase measured on these faces (the series' "measured on
    MNIST: ..." is for the digits)."""
    return TAG + what


def about(what):
    """The evidence tag of a phase that says what the data is, not what it measured."""
    return ABOUT + what


def is_tag(text):
    return text.startswith((TAG, ABOUT))


def pct(share, places=0):
    """``67%``."""
    return "%.*f%%" % (places, 100 * share)


def dist(d):
    """A distance on screen: ``0.18``."""
    return "%.2f" % float(d)


@functools.lru_cache(maxsize=1)
def vectors(name="trained"):
    """The 400 photos' unit vectors: the trained encoder's, or raw pixels'."""
    if name == "pixels":
        return F.unit(F.photos()[0].reshape(400, -1))
    return F.embed(F.rec_params(), F.small())


def gallery_ids():
    """The enrolled photos: ``v[:50:10]`` of the test people, photos 300, 310, ..., 340."""
    who = F.photos()[1]
    return np.array([np.nonzero(who == p)[0][0] for p in F.ENROLLED])


def probe_ids():
    """The 45 probes: the enrolled people's other photos, in order."""
    who = F.photos()[1]
    return np.array([q for p in F.ENROLLED for q in np.nonzero(who == p)[0][1:]])


def stranger_ids():
    who = F.photos()[1]
    return np.array([q for p in F.STRANGERS for q in np.nonzero(who == p)[0]])


def to_gallery(ids, name="trained"):
    """``torch.cdist(v[ids], gallery)``: ``[len(ids), 5]``."""
    v = vectors(name)
    return np.linalg.norm(v[ids][:, None] - v[gallery_ids()][None], axis=2)


def identify(name="trained"):
    """Region "fc-identify" on the 45 probes: per probe its photo, person, distances to the
    five enrolled faces, the nearest enrolled person, and whether that is nearer than the
    threshold (chosen on the training people)."""
    t = F.trained_threshold() if name == "trained" else F.recognition(name)["threshold"]
    ids = probe_ids()
    d = to_gallery(ids, name)
    who = F.photos()[1]
    guess = np.array(F.ENROLLED)[d.argmin(1)]
    return dict(ids=ids, person=who[ids], d=d, guess=guess, right=guess == who[ids],
                nearest=d.min(1), sure=d.min(1) < t, threshold=t)


def followed():
    """The two probes followed, each picked by a rule: ``(sure, unsure)``, the indices
    among the 45 of the first whose nearest enrolled face is nearer than the threshold,
    and of the first whose nearest is not."""
    s = identify()["sure"]
    return int(np.argmax(s)), int(np.argmax(~s))


# ====================================================================== 249
def identification():
    r = F.recognition()
    px = F.recognition("pixels")
    idn = identify()
    t = float(idn["threshold"])
    if abs(t - r["threshold"]) > 1e-6:
        raise ValueError("the threshold is not the one chosen on the training people")
    right, sure = idn["right"], idn["sure"]
    if (int(right.sum()), int((~sure).sum()), int((right & sure).sum())) != (
            r["identified"], r["unsure"], r["identified_sure"]):
        raise ValueError("identification does not match study_faces.recognition()")
    pix = identify("pixels")
    if int(pix["right"].sum()) != px["identified"]:
        raise ValueError("raw pixels' identification does not match study_faces")
    sid = stranger_ids()
    sd = to_gallery(sid)
    near = sd.min(1)
    let = near < t
    if int(let.sum()) != r["let_in"]:
        raise ValueError("strangers let in do not match study_faces.recognition()")
    who = F.photos()[1]
    x = F.photos()[0]
    gal = gallery_ids()
    a, b = followed()

    def probe(i):
        return dict(index=i, photo=int(idn["ids"][i]), person=int(idn["person"][i]),
                    image=x[idn["ids"][i]].tolist(), d=[float(v) for v in idn["d"][i]],
                    d_text=[dist(v) for v in idn["d"][i]], guess=int(idn["guess"][i]),
                    nearest=float(idn["nearest"][i]), nearest_text=dist(idn["nearest"][i]),
                    sure=bool(idn["sure"][i]), right=bool(idn["right"][i]))
    order = np.argsort(near, kind="stable")
    grid = np.array([[1.0 if right[p * 9 + j] else 0.0 for j in range(9)] for p in range(5)])
    meta = dict(
        source=SOURCE, evidence=TAG + "the ten unseen people",
        enrolled=list(F.ENROLLED), strangers_people=list(F.STRANGERS),
        gallery=[int(g) for g in gal], gallery_images=[x[g].tolist() for g in gal],
        gallery_people=[int(who[g]) for g in gal],
        gallery_vectors=vectors()[gal].tolist(), dims=int(vectors().shape[1]),
        threshold=t, threshold_text=dist(t),
        sure_probe=probe(a), unsure_probe=probe(b),
        probes=int(len(idn["ids"])), probe_ids=[int(i) for i in idn["ids"]],
        probe_person=[int(p) for p in idn["person"]], guess=[int(g) for g in idn["guess"]],
        right=[bool(v) for v in right], sure=[bool(v) for v in sure],
        nearest=[float(v) for v in idn["nearest"]], grid=grid.tolist(),
        identified=int(right.sum()), identified_sure=int((right & sure).sum()),
        unsure=int((~sure).sum()), unsure_right=int((right & ~sure).sum()),
        sure_wrong=int((~right & sure).sum()),
        pixels_identified=int(pix["right"].sum()), pixels_unsure=int((~pix["sure"]).sum()),
        both_right=int((right & pix["right"]).sum()),
        eer_text=pct(r["test_eer"], 1), pixels_eer_text=pct(px["test_eer"], 1),
        stranger_ids=[int(i) for i in sid], stranger_near=[float(v) for v in near],
        stranger_sorted=[float(v) for v in near[order]],
        stranger_nearest_person=[int(F.ENROLLED[j]) for j in sd.argmin(1)],
        stranger_count=int(len(sid)), let_in=int(let.sum()),
        let_in_ids=[int(i) for i in sid[let]],
        stranger_lo_text=dist(near.min()), stranger_hi_text=dist(near.max()),
        d_max=float(max(idn["d"].max(), near.max())),
        code_identify=list(code_lines("fc-identify")),
        code_strangers=list(code_lines("fc-strangers")),
        limitations=[
            "one enrolled photo per person, people 30-34 only; 45 probes and 50 strangers' "
            "photos: small counts, one run",
            "the encoder's nearest is right for %d of 45, exactly as many as raw pixels' "
            "(%d; %d probes are right both ways): it helps verification here (equal error "
            "rate %s against %s), not this identification" % (
                int(right.sum()), int(pix["right"].sum()), int((right & pix["right"]).sum()),
                pct(r["test_eer"], 1), pct(px["test_eer"], 1)),
            "the threshold (%.4f) was chosen on the training people's pairs, never on these "
            "people" % t,
            "the two probes followed are picked by rules: the first of the 45 whose nearest "
            "is under the threshold, and the first whose nearest is not",
            "%d probe is named wrong although it is under the threshold"
            % int((~right & sure).sum())])
    panels = (
        Panel("The gallery", "the five enrolled photos, one each of people 30-34", "images",
              dict(values=np.array([x[g] for g in gal]),
                   labels=["person %d" % p for p in F.ENROLLED])),
        Panel("One probe", "photo %d (person %d): its distance to each enrolled face"
              % (idn["ids"][a], idn["person"][a]), "bars",
              dict(values=np.array(idn["d"][a], float),
                   labels=["person %d" % p for p in F.ENROLLED])),
        Panel("All 45 probes", "1: the nearest enrolled face is the right person; rows: "
              "people 30-34, columns: their other nine photos", "matrix",
              dict(values=grid)),
        Panel("Strangers", "the 50 photos of people 35-39: the distance to the nearest "
              "enrolled face, sorted; under %.2f they are let in" % t, "bars",
              dict(values=near[order].astype(float),
                   labels=[str(int(i)) for i in sid[order]])),
    )
    return Study("Who is it?", "Five people enrolled with one photo each: 36 of their 45 "
                 "other photos named right (raw pixels: 36 too); 6 of 50 strangers let in",
                 panels, meta)


# ====================================================================== 250
def crop_of(canvas, box):
    """Region "fc-pipeline": the box's pixels (corners, clipped to the scene) resized to
    32 x 32 by area."""
    x1, y1, x2, y2 = (int(b) for b in np.clip(np.round(box), 0, F.SIDE))
    return F.resize_rect(canvas[y1:y2, x1:x2], 32)


def code_of(crop):
    """``F.normalize(encoder(crop))``: the crop's 32 numbers."""
    return F.unit(F.rec_forward(F.rec_params(), crop[None, None]))[0]


@functools.lru_cache(maxsize=1)
def scene_rank():
    """How many seeded scenes :func:`~voxanim.study_faces.pipeline_scene` drew before the
    first with three faces (the rule that picked it), and their face counts."""
    rng = np.random.default_rng([F.SEED, 6])
    _gal, pool = F._enrolled()
    counts = []
    while True:
        sc = F.scene(pool, rng)
        counts.append(int(sc["faces"]))
        if sc["faces"] == 3:
            return len(counts), counts


def pipeline():
    ps = F.pipeline_scene()
    sc = ps["scene"]
    rates = F.pipeline_rates()
    t = F.trained_threshold()
    gal = gallery_ids()
    v = vectors()
    x = F.photos()[0]
    who = F.photos()[1]
    truth = F.corners(sc["boxes"])
    faces = []
    for n in ps["named"]:
        crop = crop_of(sc["canvas"], n["box"])
        if not np.allclose(crop, n["crop"], atol=1e-6):
            raise ValueError("the crop is not pipeline_scene()'s")
        q = code_of(crop)
        d = np.linalg.norm(q[None] - v[gal], axis=1)
        if int(F.ENROLLED[int(d.argmin())]) != n["person"] or abs(d.min() - n["distance"]) > 1e-5:
            raise ValueError("the match is not pipeline_scene()'s")
        k = int(F.iou(np.asarray(n["box"], float)[None], truth)[0].argmax())
        tx, ty, ts = sc["squares"][k]
        true_crop = F.resize_rect(sc["canvas"][ty:ty + ts, tx:tx + ts], 32)
        tq = code_of(true_crop)
        td = np.linalg.norm(tq[None] - v[gal], axis=1)
        faces.append(dict(
            box=[int(b) for b in n["box"]], score=0.0, crop=crop.tolist(), q=q.tolist(),
            d=[float(e) for e in d], person=int(n["person"]), distance=float(n["distance"]),
            distance_text=dist(n["distance"]), sure=bool(n["sure"]), iou=float(n["iou"]),
            iou_text="%.2f" % n["iou"], truth=int(n["truth"]),
            right=bool(n["person"] == n["truth"]),
            true_box=[int(tx), int(ty), int(tx + ts), int(ty + ts)], true_crop=true_crop.tolist(),
            true_person=int(F.ENROLLED[int(td.argmin())]), true_distance=float(td.min()),
            size=int(n["box"][2] - n["box"][0]), true_size=int(ts)))
    for f, s in zip(faces, ps["found"]["scores"], strict=True):
        f["score"] = float(s)
        f["score_text"] = "%.2f" % s
    rank, counts = scene_rank()
    found_share = rates["found"] / rates["faces"]
    det_share = rates["right_detected"] / rates["found"]
    true_share = rates["right_true"] / rates["faces"]
    off = max(faces, key=lambda f: -f["iou"])          # the box furthest from the truth
    meta = dict(
        source=SOURCE, evidence=TAG + "the enrolled people 30-34",
        canvas=sc["canvas"].tolist(), squares=[list(map(int, s)) for s in sc["squares"]],
        faces_in_scene=int(sc["faces"]), not_faces=int(len(sc["squares"]) - sc["faces"]),
        photo_ids=[int(i) for i in sc["ids"][:sc["faces"]]], truth_people=ps["truth"],
        true_boxes=[[float(b) for b in c] for c in truth],
        found=[[float(b) for b in c] for c in ps["found"]["found"]],
        before=int(len(ps["found"]["before"])), faces=faces,
        scene_rank=rank, earlier_faces=counts[:-1],
        threshold=float(t), threshold_text=dist(t),
        gallery=[int(g) for g in gal], gallery_images=[x[g].tolist() for g in gal],
        gallery_people=[int(who[g]) for g in gal],
        named_right=int(sum(f["right"] for f in faces)),
        unsure=int(sum(not f["sure"] for f in faces)),
        off_index=faces.index(off),
        rates=rates, scenes=int(rates["scenes"]), all_faces=int(rates["faces"]),
        all_faces_text=commas(rates["faces"]), found_count=int(rates["found"]),
        right_detected=int(rates["right_detected"]), right_true=int(rates["right_true"]),
        found_share=found_share, det_share=det_share, true_share=true_share,
        found_text=pct(found_share), det_text=pct(det_share), true_text=pct(true_share),
        mean_iou=float(rates["mean_iou"]), mean_iou_text="%.2f" % rates["mean_iou"],
        code_pipeline=list(code_lines("fc-pipeline")),
        limitations=[
            "the scene is the first seeded scene of people 30-34 with three faces (scene %d "
            "of that seeded run; the earlier ones had %s faces), picked by that rule, not by "
            "how well it goes; the rates are over 200 other seeded scenes"
            % (rank, " and ".join(str(c) for c in counts[:-1])),
            "the scenes are made: photos pasted onto textured paper, so the true boxes are "
            "exact; never the enrolled photo itself",
            "named right from the detector's box: %d of the %d faces found (%s); from the "
            "true box: %d of all %d faces (%s); the two shares have different denominators"
            % (rates["right_detected"], rates["found"], pct(det_share), rates["right_true"],
               rates["faces"], pct(true_share)),
            "boxes found overlap the truth by %.2f on average (IoU); a box a little off "
            "changes the crop" % rates["mean_iou"],
            "one detector and one recogniser, each trained once; counted errors, not speeds"])
    panels = (
        Panel("The scene", "the first seeded scene of people 30-34 with three faces", "images",
              dict(values=np.array([sc["canvas"]]), labels=["scene"])),
        Panel("Cut out", "each face the detector found, cut out and resized to 32 x 32",
              "images", dict(values=np.array([f["crop"] for f in faces]),
                             labels=["person %d" % f["person"] for f in faces])),
        Panel("32 numbers each", "the encoder's unit vector of each crop", "matrix",
              dict(values=np.array([f["q"] for f in faces]))),
        Panel("200 scenes", "faces found; named right from the detector's box (of those "
              "found) and from the true box (of all)", "bars",
              dict(values=np.array([found_share, det_share, true_share]),
                   labels=["found", "named right: detector's box",
                           "named right: true box"])),
    )
    return Study("Find, cut out, compare", "On 200 scenes, 341 of 391 faces found; named "
                 "right from the detector's box 67% of those, from the true box 82%",
                 panels, meta)


# ====================================================================== 251
PEOPLE_TEST = tuple(range(F.TRAIN_PEOPLE, 40))
CANNOT = ("ages", "skin tones", "lighting", "cameras", "poses")


def limits():
    r = F.recognition()
    frr = np.asarray(r["per_person_frr"], float)
    far = np.asarray(r["per_person_far"], float)
    same_each = 45                           # 10 photos: 10 x 9 / 2 pairs of one person
    other_each = 10 * 90                     # 10 photos against the other 90
    rejects = np.round(frr * same_each).astype(int)
    accepts = np.round(far * other_each).astype(int)
    if int(rejects.sum()) != r["false_reject"] or int(accepts.sum()) != 2 * r["false_accept"]:
        raise ValueError("the per-person rates do not add up to the totals")
    who = F.photos()[1]
    if any(int((who == p).sum()) != 10 for p in range(40)):
        raise ValueError("not ten photos of each person")
    worst = int(np.argmax(frr))
    meta = dict(
        source=SOURCE, evidence=TAG + "the ten unseen people",
        people=list(PEOPLE_TEST), frr=frr.tolist(), far=far.tolist(),
        frr_text=[pct(v) for v in frr], far_text=[pct(v, 1) for v in far],
        rejects=rejects.tolist(), accepts=accepts.tolist(), same_each=same_each,
        other_each=other_each, threshold=float(r["threshold"]),
        threshold_text=dist(r["threshold"]),
        false_reject=int(r["false_reject"]), same_pairs=int(r["same_pairs"]),
        false_accept=int(r["false_accept"]), diff_pairs=int(r["pairs"] - r["same_pairs"]),
        pairs=int(r["pairs"]), pairs_text=commas(r["pairs"]),
        diff_pairs_text=commas(r["pairs"] - r["same_pairs"]),
        frr_all=r["false_reject"] / r["same_pairs"],
        frr_all_text=pct(r["false_reject"] / r["same_pairs"]),
        far_all=r["false_accept"] / (r["pairs"] - r["same_pairs"]),
        far_all_text=pct(r["false_accept"] / (r["pairs"] - r["same_pairs"]), 1),
        worst=PEOPLE_TEST[worst], worst_text=pct(frr[worst]), best_count=int((frr == 0).sum()),
        far_lo_text=pct(far.min(), 1), far_hi_text=pct(far.max(), 1),
        people_all=40, photos_each=10, photos=400, train_people=F.TRAIN_PEOPLE,
        test_people=40 - F.TRAIN_PEOPLE, lab="AT&T Laboratories Cambridge", years="1992-94",
        side=64, cannot=list(CANNOT),
        code_verify=list(code_lines("fc-verify")),
        limitations=[
            "each person's false-reject rate comes from 45 pairs and their false-accept rate "
            "from 900: one run, ten people; the differences between them are measured, not "
            "explained",
            "people are shown by number only; nothing is said about who they are",
            "40 people, ten photos each, taken at one lab in 1992-94 for research: this data "
            "cannot measure how a system errs across ages, skin tones, lighting, cameras or "
            "poses it never saw",
            "real systems are tested on far larger and more varied sets, and still err "
            "unevenly; nothing here transfers to them"])
    panels = (
        Panel("False rejects, person by person", "the share of each unseen person's 45 "
              "same-person pairs rejected at the threshold", "bars",
              dict(values=frr, labels=["person %d" % p for p in PEOPLE_TEST])),
        Panel("False accepts, person by person", "the share of each unseen person's 900 "
              "pairs with someone else accepted", "bars",
              dict(values=far, labels=["person %d" % p for p in PEOPLE_TEST])),
        Panel("Rejected pairs, counted", "each person's same-person pairs rejected, of 45",
              "bars", dict(values=rejects.astype(float),
                           labels=["person %d" % p for p in PEOPLE_TEST])),
        Panel("The whole dataset", "40 people x 10 photos; 1: trained on, 0: only tested",
              "matrix", dict(values=np.array([[1.0 if p < F.TRAIN_PEOPLE else 0.0] * 10
                                              for p in range(40)]))),
    )
    return Study("What 40 people can't tell you", "Per person, 0% to 73% of same-person "
                 "pairs rejected: and 40 people from one lab cannot show how a system errs "
                 "across people it never saw", panels, meta)


STUDIES = {"faces-identification": identification, "faces-the-whole-pipeline": pipeline,
           "faces-what-40-people-cannot-show": limits}

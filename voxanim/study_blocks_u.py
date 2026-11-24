"""The numbers behind animations 241-244 of "Blocks and code" (group ``u``, the second half of
"Finding faces: bounding boxes"): the detector's two losses, its training, one box per face,
and how good it is. Everything comes from :mod:`voxanim.study_faces`, the numpy twin of
``examples/faces.py``, trained once.

* ``faces-two-losses``: training step 0's 16 made scenes (faces of people 0-29 pasted on
  textured paper) give 16 x 144 = 2,304 cells, of which 38 hold a face's centre (1.6%).
  "there" (region "fc-loss") is binary cross-entropy over every cell, face cells counted 10
  times (region "fc-weight"): 380 of the 2,646 counted, 14.4%. "where" is the box error on
  the 38 face cells only. With the starting weights: there 0.853 (0.760 unweighted), where
  0.438, loss 0.853 + 5 x 0.438 = 3.044. The measured answer to "why the weight": the same
  run without it, on 200 scenes of the unseen people, AP 0.47 (72 of 412 faces found at 0.5)
  against 0.92 (348).
* ``faces-training-the-detector``: the two losses over the 6,000 steps (each point the mean
  of 100 steps), and held-out scene 0 (the first test scene) at steps 0, 100, 500, 2,000 and
  6,000: every cell just over 0.5 at step 0, none at 100 and 500, its two faces lit by 2,000
  (with one box too many), two boxes on its two faces at 6,000.
* ``faces-nms``: the same scene through the trained detector (region "fc-decode"): 4 cells
  above 0.5, decoded to 4 boxes, three on one face; NMS at IoU 0.3 (region "fc-nms") keeps
  the surest, 0.98, and drops 0.96 (IoU 0.77 with it) and 0.75 (0.39): 2 boxes for 2 faces.
  The scene is picked by a rule: the first test scene where NMS drops a box (scene 0). Over
  the 200 test scenes, 685 boxes above 0.5 before NMS, 412 after.
* ``faces-precision-and-recall``: matching at IoU 0.5 on the first test scene with a wrong
  box (scene 3: 3 right, 1 wrong); over the 200 test scenes at 0.5, 348 of the 412 faces
  found and 64 wrong boxes (precision and recall both 84.5%: 412 boxes for 412 faces, by
  chance); precision against recall as the confidence falls to 0.05, AP (the area under it)
  0.92; without the weight, 0.47.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_faces as F
from .blocks import code_lines, commas, shape
from .study import Panel, Study

PW, PW_OFF = 10.0, 1.0          # the script's weight on face cells, and no weight
BATCH_STEP = 0                  # 241: the first training step's 16 scenes
THUMB = 48                      # 241: the scenes drawn small, area-resized 96 -> 48
WINDOW = 100                    # 242: each point of the curves is the mean of 100 steps
HELD_OUT = 0                    # 242: the first test scene
KEEPS = (0.9, 0.7, 0.5, 0.3, 0.1)   # 244: confidences marked along the curve
SOURCE = ("the Olivetti faces (AT&T Laboratories Cambridge, 1992-94): made scenes of faces "
          "pasted on textured paper; the detector trained once as examples/faces.py trains "
          "it (voxanim.study_faces, its numpy twin); tested on people 30-39, never seen")


def pct(share, places=1):
    """``84.5%``."""
    return "%.*f%%" % (places, 100 * share)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def three(v):
    return "%.3f" % v


def two(v):
    return "%.2f" % v


def thumb(canvas, side=THUMB):
    """A scene area-resized to ``side`` (as ``F.interpolate(mode="area")``)."""
    return F.resize(np.asarray(canvas, np.float32), side)


def face_cells(boxes):
    """The ``(row, col)`` cell each face's centre falls in (``offsets``)."""
    return [F.offsets(*b)[:2] for b in boxes]


# ====================================================================== 241
@functools.lru_cache(maxsize=1)
def batch_losses():
    """Training step 0's batch through the starting weights: both parts of the loss, with
    and without the weight, and one face cell's box worked out."""
    b = F.det_batch(BATCH_STEP)
    P = F.det_params(0)
    out = F.det_forward(P, b["x"])
    loss, there, where, _d = F.det_loss(out, b["has"], b["want"], PW)
    loss1, there1, where1, _d1 = F.det_loss(out, b["has"], b["want"], PW_OFF)
    has = b["has"]
    # one face cell: the first face of the first scene
    box = b["scenes"][0]["boxes"][0]
    r, c, want = F.offsets(*box)
    said = np.r_[sig(out[0, 1:3, r, c]), out[0, 3:5, r, c]].astype(np.float64)
    miss = float(((said - want) ** 2).sum())
    p = sig(out[:, 0])
    return dict(batch=b, out=out, has=has, loss=loss, there=there, where=where,
                loss1=loss1, there1=there1, where1=where1, cell=(0, r, c), box=box,
                want=want.astype(np.float64), said=said, miss=miss, p=p)


def two_losses():
    L = batch_losses()
    b, has = L["batch"], L["has"]
    n_cells, n_face = int(has.size), int(has.sum())
    n_empty = n_cells - n_face
    w_face = PW * n_face
    d, d1 = F.detection(PW), F.detection(PW_OFF)
    per_scene = [int(h.sum()) for h in has]
    if per_scene != [s["faces"] for s in b["scenes"]]:
        raise ValueError("two faces share a cell")
    # BCE per cell at the start: a face cell counts -log p, ten times; an empty one -log(1-p)
    p = L["p"]
    there_check = float(-(PW * has * np.log(p) + (1 - has) * np.log(1 - p)).mean())
    if abs(there_check - L["there"]) > 1e-4:
        raise ValueError("there is not the weighted mean over every cell")
    if abs(L["loss"] - (L["there"] + F.WHERE_WEIGHT * L["where"])) > 1e-6:
        raise ValueError("the loss is not there + 5 x where")
    s, r, c = L["cell"]
    meta = dict(
        source=SOURCE, evidence="measured on the Olivetti faces: training step 0's batch "
        "through the starting weights; the two trained runs on 200 test scenes",
        step=BATCH_STEP, scenes=len(has), grid=F.GRID, cells=n_cells, cells_text=commas(n_cells),
        per_scene=per_scene, face_cells=n_face, empty_cells=n_empty,
        empty_text=commas(n_empty), face_share=n_face / n_cells,
        face_share_text=pct(n_face / n_cells),
        weight=PW, weight_text="%d" % PW, weighted_face=w_face,
        weighted_face_text=commas(w_face), weighted_total=w_face + n_empty,
        weighted_total_text=commas(w_face + n_empty),
        weighted_share=w_face / (w_face + n_empty),
        weighted_share_text=pct(w_face / (w_face + n_empty)),
        has=has.astype(int).tolist(),
        thumbs=[np.round(thumb(x[0]), 3).tolist() for x in b["x"]], thumb_side=THUMB,
        p_lo=float(p.min()), p_hi=float(p.max()), p_lo_text=two(p.min()), p_hi_text=two(p.max()),
        there=L["there"], there_text=three(L["there"]),
        there_unweighted=L["there1"], there_unweighted_text=three(L["there1"]),
        where=L["where"], where_text=three(L["where"]),
        where_weight=F.WHERE_WEIGHT, loss=L["loss"], loss_text=three(L["loss"]),
        cell=dict(scene=s, row=r, col=c, box=[float(v) for v in L["box"]],
                  want=L["want"].tolist(), said=L["said"].tolist(),
                  want_text=[two(v) for v in L["want"]], said_text=[two(v) for v in L["said"]],
                  miss=L["miss"], miss_text=three(L["miss"])),
        ap=d["ap"], ap_text=two(d["ap"]), ap_unweighted=d1["ap"],
        ap_unweighted_text=two(d1["ap"]), faces=d["faces"], found=d["found"],
        found_unweighted=d1["found"], wrong=d["wrong"], wrong_unweighted=d1["wrong"],
        test_scenes=F.TEST_SCENES,
        out_shape=shape(L["out"][:, 0].shape),
        code_loss=list(code_lines("fc-loss")), code_weight=list(code_lines("fc-weight")),
        limitations=[
            "the scenes are made: faces pasted on textured paper, with not-faces (a photo "
            "shuffled in 8 x 8 blocks); the boxes are exact because they were pasted",
            "the losses shown are one batch (training step 0) through the starting weights",
            "weighted against unweighted: one run each, from the same start on the same "
            "scenes; tested on 200 scenes of people 30-39, never seen in training",
            "AP at IoU 0.5 over boxes kept from 0.05 up; 'found' at the script's 0.5"])
    panels = (
        Panel("Face cells of one batch", "the 16 scenes' 12 x 12 grids side by side: 1 where "
              "a face's centre is", "matrix",
              dict(values=has.reshape(4, 4, 12, 12).transpose(0, 2, 1, 3).reshape(48, 48)
                   .astype(float))),
        Panel("What 'there' counts", "cells of one batch, and their weight in 'there'",
              "bars", dict(values=np.array([n_face, n_empty, w_face, n_empty], float),
                           labels=["face cells", "empty cells", "face cells x 10",
                                   "empty cells x 1"])),
        Panel("One face cell's box", "what the cell must say and what it says at the start",
              "bars", dict(values=np.r_[L["want"], L["said"]],
                           labels=["want dx", "want dy", "want log w/16", "want log h/16",
                                   "says dx", "says dy", "says log w/16", "says log h/16"])),
        Panel("Why the weight", "AP at IoU 0.5 on 200 test scenes, face cells weighted 10 "
              "times and not weighted", "bars",
              dict(values=np.array([d["ap"], d1["ap"]]), labels=["weighted", "unweighted"])),
    )
    return Study("Two losses: is it there, and where", "There: every cell, face cells x 10 "
                 "(38 of 2,304 cells hold a face). Where: face cells only. Unweighted, AP "
                 "falls from 0.92 to 0.47", panels, meta)


# ====================================================================== 242
def smoothed(v, window=WINDOW):
    """Step 0's value, then the means of consecutive ``window`` steps at their middles."""
    v = np.asarray(v, np.float64)
    n = len(v) // window
    xs = np.r_[0.0, np.arange(n) * window + window / 2.0]
    return xs, np.r_[v[0], v[:n * window].reshape(n, window).mean(1)]


@functools.lru_cache(maxsize=4)
def snapshot(step, index=HELD_OUT, pw=PW):
    """Test scene ``index`` through the detector at ``step``: its grid of "a face here?",
    the cells above 0.5, and the boxes kept after NMS."""
    sc = F.test_scenes()[index]
    P = F.det_params(step, pw)
    out = F.det_forward(P, sc["canvas"][None, None])[0]
    p = sig(out[0])
    f = F.find(P, sc["canvas"])
    truth = F.corners(sc["boxes"])
    o = F.iou(f["found"], truth) if len(f["found"]) else np.zeros((0, len(truth)))
    right = [bool(row.max() >= F.MATCH_IOU) for row in o]
    return dict(step=step, p=p, above=int((p > F.KEEP).sum()), p_max=float(p.max()),
                p_min=float(p.min()), found=f["found"], scores=f["scores"], right=right,
                face_p=[float(p[r, c]) for r, c in face_cells(sc["boxes"])])


def training():
    run = F.det_run(PW)
    xs, there = smoothed(run["there"])
    _, where = smoothed(run["where"])
    sc = F.test_scenes()[HELD_OUT]
    snaps = [snapshot(s) for s in F.DET_SNAPS]
    cells = face_cells(sc["boxes"])
    last = snaps[-1]
    meta = dict(
        source=SOURCE, evidence="measured on the Olivetti faces: one run of 6,000 steps; "
        "held-out test scene 0",
        steps=F.DET_STEPS, steps_text=commas(F.DET_STEPS), batch=F.BATCH, window=WINDOW,
        curve_x=xs.tolist(), there=there.tolist(), where=where.tolist(),
        there_first=float(run["there"][0]), where_first=float(run["where"][0]),
        there_first_text=two(run["there"][0]), where_first_text=two(run["where"][0]),
        there_last=float(there[-1]), where_last=float(where[-1]),
        there_last_text=two(there[-1]), where_last_text=two(where[-1]),
        loss_first_text=two(run["loss"][0]), loss_last_text=two(run["loss"][-WINDOW:].mean()),
        scene=HELD_OUT, canvas=sc["canvas"].tolist(), faces=sc["faces"],
        not_faces=len(sc["squares"]) - sc["faces"],
        truth=F.corners(sc["boxes"]).tolist(), face_cells=[list(rc) for rc in cells],
        people=[int(F.photos()[1][i]) for i in sc["ids"][:sc["faces"]]],
        snaps=[dict(step=s["step"], step_text=commas(s["step"]), p=s["p"].tolist(),
                    above=s["above"], p_max=s["p_max"], p_min=s["p_min"],
                    p_max_text=two(s["p_max"]), p_min_text=two(s["p_min"]),
                    found=s["found"].tolist(), scores=s["scores"].tolist(),
                    score_texts=[two(v) for v in s["scores"]], right=s["right"],
                    boxes=len(s["found"]), right_count=int(sum(s["right"])),
                    face_p=s["face_p"], face_p_text=[two(v) for v in s["face_p"]])
               for s in snaps],
        code_train=list(code_lines("fc-train")), code_decode=list(code_lines("fc-decode")),
        limitations=[
            "one training run; the curves start at step 0's value, then each point is the "
            "mean of 100 steps' batches, 16 new made scenes a step",
            "the scene followed is the first of the 200 test scenes (people 30-39), picked "
            "by that rule, not for how it looks",
            "at step 0 every cell says just over 0.5 (%s to %s): the untrained detector's "
            "boxes are not drawn" % (two(snaps[0]["p_min"]), two(snaps[0]["p_max"])),
            "boxes drawn are those kept by the script's find: above 0.5, NMS at IoU 0.3"])
    if not (len(last["found"]) == sc["faces"] == sum(last["right"])):
        raise ValueError("the trained detector does not find this scene's faces")
    panels = (
        Panel("The two losses", "there and where over 6,000 steps, each point the mean of "
              "100 steps", "curve", dict(x=xs, ys=np.array([there, where]),
                                         labels=["there", "where"])),
        Panel("A held-out scene", "test scene 0: %d faces of unseen people and %d not-faces"
              % (sc["faces"], len(sc["squares"]) - sc["faces"]), "images",
              dict(values=sc["canvas"][None].astype(float), labels=["test scene 0"])),
        Panel("Its grid at step 6,000", "how sure each cell is that a face's centre is in it",
              "matrix", dict(values=last["p"])),
        Panel("Cells above 0.5", "cells of the held-out scene above 0.5, by step", "bars",
              dict(values=np.array([s["above"] for s in snaps], float),
                   labels=[commas(s) for s in F.DET_SNAPS])),
    )
    return Study("Training the detector", "6,000 steps: there falls from 0.85 to 0.06, where "
                 "from 0.44 to 0.03; a held-out scene's faces light up", panels, meta)


# ====================================================================== 243
def nms_scene():
    """The scene of "One box per face": the first test scene where NMS drops a box."""
    P = F.det_params()
    for i, sc in enumerate(F.test_scenes()):
        f = F.find(P, sc["canvas"])
        if len(f["found"]) < len(f["before"]):
            return i
    raise ValueError("NMS drops nothing on any test scene")


@functools.lru_cache(maxsize=1)
def nms_totals():
    """Boxes above 0.5 before and after NMS over the 200 test scenes."""
    P = F.det_params()
    before = after = 0
    for sc in F.test_scenes():
        f = F.find(P, sc["canvas"])
        before += len(f["before"])
        after += len(f["found"])
    return before, after


def one_box():
    i = nms_scene()
    sc = F.test_scenes()[i]
    P = F.det_params()
    out = F.det_forward(P, sc["canvas"][None, None])[0]
    p = sig(out[0])
    boxes, scores, cells = F.decode(out)
    cor = F.corners(boxes)
    keep = F.nms(cor, scores)
    o = F.iou(cor, cor)
    truth = F.corners(sc["boxes"])
    ot = F.iou(cor, truth)
    order = [int(j) for j in np.argsort(-scores, kind="stable")]
    # who drops whom: each dropped box, the kept box it overlaps most (above 0.3)
    fate = []
    for j in range(len(scores)):
        if j in keep:
            fate.append(dict(kept=True, by=None, iou=None))
        else:
            k = max((int(k) for k in keep if o[k, j] > F.NMS_IOU), key=lambda k: scores[k])
            fate.append(dict(kept=False, by=k, iou=float(o[k, j]), iou_text=two(o[k, j])))
    # the surest cell, worked: logits -> pixels
    top = order[0]
    r, c = (int(v) for v in cells[top])
    sx, sy = sig(out[1:3, r, c])
    lw, lh = (float(v) for v in out[3:5, r, c])
    before, after = nms_totals()
    groups = {}
    for j in range(len(scores)):
        groups.setdefault(int(ot[j].argmax()) if ot[j].max() > 0 else -1, []).append(j)
    face = max(groups, key=lambda g: len(groups[g]))           # the face claimed most
    meta = dict(
        source=SOURCE, evidence="measured on the Olivetti faces: test scene %d through the "
        "trained detector" % i,
        scene=i, canvas=sc["canvas"].tolist(), faces=sc["faces"],
        not_faces=len(sc["squares"]) - sc["faces"], truth=truth.tolist(),
        face_cells=[list(rc) for rc in face_cells(sc["boxes"])],
        p=p.tolist(), keep_at=F.KEEP, nms_iou=F.NMS_IOU, nms_iou_text="%.1f" % F.NMS_IOU,
        cells=[[int(a), int(b)] for a, b in cells], scores=[float(v) for v in scores],
        score_texts=[two(v) for v in scores], order=order,
        boxes=boxes.tolist(), corners=cor.tolist(),
        box_texts=["(%.0f, %.0f, %.0f, %.0f)" % tuple(b) for b in boxes],
        iou=o.tolist(), iou_truth=ot.tolist(),
        kept=[int(k) for k in keep], fate=fate, n_before=len(scores), n_after=len(keep),
        kept_iou_truth=[float(ot[k].max()) for k in keep],
        kept_iou_truth_text=[two(ot[k].max()) for k in keep],
        claimed=groups[face], claimed_face=face,
        worked=dict(index=top, row=r, col=c, sx=float(sx), sy=float(sy), lw=lw, lh=lh,
                    sx_text=two(sx), sy_text=two(sy), cx=float((c + sx) * F.CELL),
                    cy=float((r + sy) * F.CELL), w=16 * float(np.exp(lw)),
                    h=16 * float(np.exp(lh)),
                    cx_text="%.1f" % ((c + sx) * F.CELL), cy_text="%.1f" % ((r + sy) * F.CELL),
                    w_text="%.1f" % (16 * np.exp(lw)), h_text="%.1f" % (16 * np.exp(lh)),
                    score_text=two(scores[top])),
        total_before=before, total_after=after, total_scenes=F.TEST_SCENES,
        code_decode=list(code_lines("fc-decode")), code_nms=list(code_lines("fc-nms")),
        limitations=[
            "the scene is picked by a rule: the first test scene where NMS drops a box "
            "(test scene %d)" % i,
            "boxes above 0.5 only (the script's keep); NMS at IoU 0.3, as torchvision's nms",
            "over all 200 test scenes, %d boxes above 0.5 before NMS and %d after"
            % (before, after)])
    panels = (
        Panel("The scene", "test scene %d: %d faces of unseen people, %d not-faces"
              % (i, sc["faces"], len(sc["squares"]) - sc["faces"]), "images",
              dict(values=sc["canvas"][None].astype(float), labels=["test scene %d" % i])),
        Panel("How sure each cell is", "the sigmoid of each cell's first number", "matrix",
              dict(values=p)),
        Panel("Boxes above 0.5", "the decoded boxes' scores", "bars",
              dict(values=np.asarray(scores, float),
                   labels=["cell (%d, %d)" % (a, b) for a, b in cells])),
        Panel("How much they overlap", "IoU of every pair of decoded boxes", "matrix",
              dict(values=o)),
    )
    return Study("One box per face", "Four boxes above 0.5, three on one face; NMS at IoU "
                 "0.3 keeps the surest of each and drops the rest: 2 boxes for 2 faces",
                 panels, meta)


# ====================================================================== 244
def wrong_scene():
    """The scene of the matching: the first test scene with a wrong box at 0.5."""
    P = F.det_params()
    for i, sc in enumerate(F.test_scenes()):
        f = F.find(P, sc["canvas"])
        ok, _ = F._match(f["found"], f["scores"], sc["boxes"])
        if not all(g for _s, g in ok):
            return i
    raise ValueError("no test scene has a wrong box")


def at_keep(d, keep):
    """Recall and precision with the boxes from confidence ``keep`` up."""
    n = int(np.searchsorted(-d["scores"], -keep, side="right"))
    if n == 0:
        return 0.0, 1.0, 0
    return float(d["recall"][n - 1]), float(d["precision"][n - 1]), n


def area(d):
    """AP as the script's evaluation adds it up: precision times each step of recall."""
    r = np.asarray(d["recall"], np.float64)
    p = np.asarray(d["precision"], np.float64)
    dr = np.diff(np.r_[0.0, r])
    return float((p * dr).sum())


def curve_meta(d):
    out = dict(recall=d["recall"].astype(float).tolist(),
               precision=d["precision"].astype(float).tolist(),
               scores=d["scores"].astype(float).tolist(), ap=d["ap"], ap_text=two(d["ap"]),
               faces=d["faces"], found=d["found"], wrong=d["wrong"], on_not=d["on_not"],
               mean_iou=d["mean_iou"], mean_iou_text=two(d["mean_iou"]))
    r, p, n = at_keep(d, F.KEEP)
    out.update(recall05=r, precision05=p, boxes05=n, recall05_text=pct(r),
               precision05_text=pct(p), last_recall=float(d["recall"][-1]),
               last_recall_text=pct(d["recall"][-1]),
               last_precision_text=pct(d["precision"][-1]),
               marks=[dict(keep=k, keep_text="%.1f" % k, recall=at_keep(d, k)[0],
                           precision=at_keep(d, k)[1], recall_text=pct(at_keep(d, k)[0]),
                           precision_text=pct(at_keep(d, k)[1])) for k in KEEPS])
    return out


def precision_recall():
    i = wrong_scene()
    sc = F.test_scenes()[i]
    f = F.find(F.det_params(), sc["canvas"])
    truth = F.corners(sc["boxes"])
    o = F.iou(f["found"], truth)
    ok, _ = F._match(f["found"], f["scores"], sc["boxes"])
    if [s for s, _g in ok] != [float(s) for s in f["scores"]]:
        raise ValueError("the kept boxes are not surest first")
    right = [g for _s, g in ok]
    d, d1 = F.detection(PW), F.detection(PW_OFF)
    w, u = curve_meta(d), curve_meta(d1)
    for m, dd in ((w, d), (u, d1)):
        if abs(area(dd) - dd["ap"]) > 1e-6:
            raise ValueError("AP is not the area under precision against recall")
        if m["boxes05"] != m["found"] + m["wrong"]:
            raise ValueError("the boxes at 0.5 are not found + wrong")
        if round(m["recall05"] * m["faces"]) != m["found"]:
            raise ValueError("recall at 0.5 is not found / faces")
    meta = dict(
        source=SOURCE, evidence="measured on the Olivetti faces: 200 test scenes of people "
        "30-39, one run each",
        scene=i, canvas=sc["canvas"].tolist(), faces_here=sc["faces"],
        not_faces=len(sc["squares"]) - sc["faces"], truth=truth.tolist(),
        found=f["found"].tolist(), scores=f["scores"].astype(float).tolist(),
        score_texts=[two(v) for v in f["scores"]],
        ious=[float(row.max()) for row in o], iou_texts=[two(row.max()) for row in o],
        right=right, right_here=int(sum(right)), wrong_here=int(len(right) - sum(right)),
        match_iou=F.MATCH_IOU, keep=F.KEEP, scenes=F.TEST_SCENES,
        weighted=w, unweighted=u,
        code_iou=list(code_lines("fc-iou")), code_decode=list(code_lines("fc-decode")),
        limitations=[
            "the scene matched is picked by a rule: the first test scene with a wrong box "
            "at 0.5 (test scene %d)" % i,
            "a box is right when its IoU with a face not already matched is at least 0.5, "
            "surest boxes first",
            "AP: precision summed over each step of recall, boxes kept from 0.05 up (so "
            "the curve ends at recall %s)" % w["last_recall_text"],
            "at 0.5, precision and recall are both %s: 412 boxes for 412 faces, by chance"
            % w["precision05_text"],
            "one run each; the unweighted run's AP is 0.47 and it finds 72 faces at 0.5"])
    panels = (
        Panel("Matching", "test scene %d: the boxes kept at 0.5 and the true faces" % i,
              "images", dict(values=sc["canvas"][None].astype(float),
                             labels=["test scene %d" % i])),
        Panel("At confidence 0.5", "200 test scenes: faces, right boxes, wrong boxes", "bars",
              dict(values=np.array([d["faces"], d["found"], d["wrong"]], float),
                   labels=["faces", "right boxes", "wrong boxes"])),
        Panel("Precision against recall", "boxes taken surest first, down to 0.05; AP %s"
              % w["ap_text"], "curve",
              dict(x=d["recall"].astype(float), ys=d["precision"][None].astype(float),
                   labels=["weighted"])),
        Panel("Without the weight", "the unweighted run's curve; AP %s" % u["ap_text"],
              "curve", dict(x=d1["recall"].astype(float), ys=d1["precision"][None].astype(float),
                            labels=["unweighted"])),
    )
    return Study("Is the detector any good?", "On 200 scenes of unseen people: 348 of 412 "
                 "faces found and 64 wrong boxes at 0.5; AP 0.92, against 0.47 without the "
                 "weight", panels, meta)


STUDIES = {"faces-two-losses": two_losses, "faces-training-the-detector": training,
           "faces-nms": one_box, "faces-precision-and-recall": precision_recall}

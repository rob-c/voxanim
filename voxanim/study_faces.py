"""Finding faces and telling them apart, for the face animations of "Blocks and code" (catalogue
238-251): ``examples/faces.py`` in numpy, trained and tested once, with every number shown.

Data: the Olivetti faces (:mod:`voxanim.olivetti`; ten photos of each of 40 people,
AT&T Laboratories Cambridge, 1992-94). People 0-29 train both models; people 30-39 are kept
for testing. Every test number is on people the models never saw.

**The detector** (:func:`det_forward`): three blocks of a 3 x 3 convolution (padding 1), ReLU
and a 2 x 2 max-pool (1 -> 8 -> 16 -> 32 channels, 96 -> 48 -> 24 -> 12), then a 1 x 1
convolution to 5 numbers per cell of the 12 x 12 grid: a logit for "a face's centre is in
this cell" and the box (the centre in the cell through a sigmoid, the width and height as
log(size / 16)). Trained on scenes (:func:`scene`): 96 x 96 grey textured paper with one to
three faces of the training people pasted in at sizes 16 to 32 and up to two not-faces (a
photo cut into 8 x 8 blocks and shuffled). The loss (:func:`det_loss`) is binary
cross-entropy over every cell, face cells weighted ``pw`` times, plus 5 x the squared error of
the box on the face cells only. Adam, 6,000 steps of 16 scenes; run twice, ``pw`` 10 (the
script's) and 1 (unweighted), from the same start on the same scenes. Tested on 200 scenes of
the test people (:func:`det_eval`): boxes decoded, duplicates removed by non-maximum
suppression at IoU 0.3, matched to the true boxes at IoU 0.5; precision, recall and AP.

**The recogniser** (:func:`rec_forward`): the same blocks on the 32 x 32 photo (2 x 2
average), flattened (512) to 32 numbers, scaled to length 1. Trained with
``F.triplet_margin_loss`` (margin 0.5, PyTorch's distance with eps 1e-6) on 32 random
triplets a step of the training people, Adam, 1,500 steps. Tested (:func:`rec_eval`) on the
100 photos of the test people: verification over their 4,950 pairs at the threshold chosen
on the training people's pairs (where false accepts and false rejects are equal),
identification of people 30-34 from one enrolled photo each, people 35-39 as strangers, and
the same numbers for raw pixels and the untrained network.

Everything is cached by :func:`voxanim.cache.cached_npz`. ``tests/test_faces_equivalence.py``
runs the script's own lines in torch from this module's starting weights on its scenes and
triplets, and matches one Adam step of each model, the decoding and NMS, and the resizing.
The script's random scenes and triplets come from torch's generator, the twin's from numpy's.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import sys

import numpy as np

from . import olivetti
from . import study_course as S
from . import study_course_cnn as C
from .cache import cached_npz

VERSION = 1
SIDE, CELL, GRID = 96, 8, 12
TRAIN_PEOPLE = 30
SIZES = (16, 32)
BATCH = 16
DET_STEPS = 6000
PWS = (10.0, 1.0)
WHERE_WEIGHT = 5.0
NMS_IOU = 0.3
MATCH_IOU = 0.5
KEEP = 0.5
TEST_SCENES = 200
DET_SNAPS = (0, 100, 500, 2000, 6000)
CHANNELS = (1, 8, 16, 32)
DET_KEYS = ("0.weight", "0.bias", "3.weight", "3.bias", "6.weight", "6.bias",
            "9.weight", "9.bias")
REC_KEYS = ("0.weight", "0.bias", "3.weight", "3.bias", "6.weight", "6.bias",
            "10.weight", "10.bias")
EMBED = 32
MARGIN = 0.5
EPS = 1e-6
TRIPLETS = 32
REC_STEPS = 1500
REC_SNAPS = (0, 100, 500, 1500)
ENROLLED = tuple(range(30, 35))
STRANGERS = tuple(range(35, 40))
SEED = 238
SCRIPT = "examples/faces.py"


# ====================================================================== the data
@functools.lru_cache(maxsize=1)
def photos():
    """``(photos [400, 64, 64] 0..1, person [400])`` as ``fetch_olivetti_faces`` gives them."""
    return olivetti.load()


@functools.lru_cache(maxsize=1)
def small():
    """``F.avg_pool2d(photos[:, None], 2)``: ``[400, 1, 32, 32]``."""
    x = photos()[0]
    return x.reshape(400, 32, 2, 32, 2).mean((2, 4), dtype=np.float32)[:, None]


@functools.cache
def _area(s, n=64):
    """The ``[s, n]`` matrix of ``F.interpolate(mode="area")`` along one axis: output ``i``
    averages inputs ``floor(i n / s)`` to ``ceil((i + 1) n / s)``."""
    A = np.zeros((s, n), np.float32)
    for i in range(s):
        a, b = (i * n) // s, -(-((i + 1) * n) // s)
        A[i, a:b] = 1.0 / (b - a)
    return A


def resize(img, s):
    """``F.interpolate(img[None, None], size=s, mode="area")[0, 0]`` (square ``img``)."""
    A = _area(s, img.shape[0])
    return (A @ img @ A.T).astype(np.float32)


def background(rng):
    """Grey paper with a soft texture: uniform noise averaged over 9 x 9 (``[96, 96]``)."""
    noise = rng.random((SIDE + 8, SIDE + 8), dtype=np.float32)
    c = np.cumsum(np.cumsum(np.pad(noise, ((1, 0), (1, 0))), 0), 1)
    return ((c[9:, 9:] - c[:-9, 9:] - c[9:, :-9] + c[:-9, :-9]) / 81).astype(np.float32)


def scramble(img, rng):
    """A not-face: the photo cut into 8 x 8 blocks, the 64 blocks shuffled."""
    b = img.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 8, 8)[rng.permutation(64)]
    return b.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 64)


def places(n, rng, lo=SIZES[0], hi=SIZES[1]):
    out = []
    while len(out) < n:
        s = int(rng.integers(lo, hi + 1))
        x, y = (int(v) for v in rng.integers(0, SIDE - s + 1, 2))
        if all(x + s <= a or a + t <= x or y + s <= b or b + t <= y for a, b, t in out):
            out.append((x, y, s))
    return out


def scene(pool, rng):
    """One scene: ``dict(canvas [96, 96], boxes [(cx, cy, w, h)], squares [(x, y, s)],
    ids, faces)`` (``faces``: how many of ``squares`` are faces; the rest are not-faces)."""
    x = photos()[0]
    nf, nn_ = int(rng.integers(1, 4)), int(rng.integers(0, 3))
    canvas = background(rng)
    ids = np.asarray(pool)[rng.integers(0, len(pool), nf + nn_)]
    sq = places(nf + nn_, rng)
    boxes = []
    for k, (px, py, s) in enumerate(sq):
        img = x[ids[k]] if k < nf else scramble(x[ids[k]], rng)
        canvas[py:py + s, px:px + s] = resize(img, s)
        if k < nf:
            boxes.append((px + s / 2, py + s / 2, float(s), float(s)))
    return dict(canvas=canvas, boxes=boxes, squares=sq, ids=ids, faces=nf)


def train_pool():
    return np.nonzero(photos()[1] < TRAIN_PEOPLE)[0]


def test_pool():
    return np.nonzero(photos()[1] >= TRAIN_PEOPLE)[0]


def offsets(x, y, w, h):
    """``offsets`` of the script: the cell ``(r, c)`` and what it must say."""
    r, c = int(y // CELL), int(x // CELL)
    return r, c, np.array([x / CELL - c, y / CELL - r, np.log(w / 16), np.log(h / 16)],
                          np.float32)


def targets(boxes):
    """``targets`` of the script: ``has [12, 12]`` and ``want [4, 12, 12]``."""
    has = np.zeros((GRID, GRID), np.float32)
    want = np.zeros((4, GRID, GRID), np.float32)
    for box in boxes:
        r, c, say = offsets(*box)
        has[r, c] = 1
        want[:, r, c] = say
    return has, want


def det_batch(step, seed=0):
    """Training step ``step``'s 16 scenes, stacked: ``dict(x [16, 1, 96, 96], has, want,
    scenes)``."""
    rng = np.random.default_rng([SEED, 1, int(seed), int(step)])
    scenes = [scene(train_pool(), rng) for _ in range(BATCH)]
    t = [targets(s["boxes"]) for s in scenes]
    return dict(x=np.stack([s["canvas"] for s in scenes])[:, None],
                has=np.stack([a for a, _ in t]), want=np.stack([b for _, b in t]),
                scenes=scenes)


@functools.lru_cache(maxsize=1)
def test_scenes(n=TEST_SCENES):
    """The ``n`` held-out scenes (people 30-39 only), seeded."""
    rng = np.random.default_rng([SEED, 2])
    return [scene(test_pool(), rng) for _ in range(n)]


# ====================================================================== the blocks
def _init(shapes, seed, stream):
    rng = np.random.default_rng([SEED, stream, int(seed)])
    P = {}
    for k, shape in shapes.items():
        bound = 1.0 / np.sqrt(np.prod(shape[1:]))
        P[k + ".weight"] = rng.uniform(-bound, bound, shape).astype(np.float32)
        P[k + ".bias"] = rng.uniform(-bound, bound, shape[0]).astype(np.float32)
    return P


def _pad1(x):
    return np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))


def _blocks(P, x, keys=("0", "3", "6")):
    """The three blocks; returns the output and what backward needs."""
    cache, h = [], x
    for k in keys:
        z = C.conv(_pad1(h), P[k + ".weight"], P[k + ".bias"])
        a = np.maximum(z, 0)
        p, where = C.pool(a)
        cache.append((h, z, a, where))
        h = p
    return h, cache


def _blocks_back(P, d, cache, keys=("0", "3", "6")):
    G = {}
    for k, (h, z, a, where) in zip(keys[::-1], cache[::-1], strict=True):
        d = C.pool_back(d, where, a.shape) * (z > 0)
        dh, G[k + ".weight"], G[k + ".bias"] = C.conv_back(d, _pad1(h), P[k + ".weight"])
        d = dh[:, :, 1:-1, 1:-1]
    return G


def count(P):
    return int(sum(v.size for v in P.values()))


# ====================================================================== the detector
def det_init(seed=0):
    return _init({"0": (8, 1, 3, 3), "3": (16, 8, 3, 3), "6": (32, 16, 3, 3),
                  "9": (5, 32, 1, 1)}, seed, 3)


def det_forward(P, x, keep_cache=False):
    """``detector(x)``: ``[B, 5, 12, 12]`` (and the cache for backward)."""
    h, cache = _blocks(P, np.asarray(x, P["0.weight"].dtype))
    W = P["9.weight"].reshape(5, 32)
    out = np.einsum("oc,bchw->bohw", W, h) + P["9.bias"][None, :, None, None]
    out = out.astype(W.dtype)
    return (out, (h, cache)) if keep_cache else out


def _sig(z):
    z = np.asarray(z)
    return (1.0 / (1.0 + np.exp(-z))).astype(z.dtype)


def det_loss(out, has, want, pw=10.0):
    """The loss of region "fc-loss" and its gradient: ``(loss, there, where, dout)``."""
    obj, box = out[:, 0], out[:, 1:]
    n = obj.size
    # binary cross-entropy with logits, face cells weighted pw times (PyTorch's form)
    log_sig = -np.logaddexp(0, -obj)
    log_one_minus = -np.logaddexp(0, obj)
    there = float(-(pw * has * log_sig + (1 - has) * log_one_minus).mean())
    s = _sig(obj)
    d_obj = ((s * (pw * has + 1 - has) - pw * has) / n).astype(out.dtype)
    xy, wh = _sig(box[:, :2]), box[:, 2:]
    faces = float(has.sum())
    miss = ((xy - want[:, :2]) ** 2).sum(1) + ((wh - want[:, 2:]) ** 2).sum(1)
    where = float((miss * has).sum() / faces)
    k = WHERE_WEIGHT / faces
    d_xy = (k * 2 * (xy - want[:, :2]) * has[:, None] * xy * (1 - xy)).astype(out.dtype)
    d_wh = (k * 2 * (wh - want[:, 2:]) * has[:, None]).astype(out.dtype)
    dout = np.concatenate([d_obj[:, None], d_xy, d_wh], 1)
    return there + WHERE_WEIGHT * where, there, where, dout


def det_loss_and_grads(P, x, has, want, pw=10.0):
    out, (h, cache) = det_forward(P, x, keep_cache=True)
    loss, there, where, dout = det_loss(out, has, want, pw)
    W = P["9.weight"].reshape(5, 32)
    G = {"9.weight": np.einsum("bohw,bchw->oc", dout, h).reshape(5, 32, 1, 1),
         "9.bias": dout.sum((0, 2, 3))}
    dh = np.einsum("oc,bohw->bchw", W, dout).astype(W.dtype)
    G.update(_blocks_back(P, dh, cache))
    return loss, there, where, {k: G[k].astype(P[k].dtype) for k in P}


def _train_det(pw):
    P = det_init()
    state = S.adam_start(P)
    out = dict(loss=[], there=[], where=[])
    for step in range(DET_STEPS):
        if step in DET_SNAPS:
            out.update({"W%d__%s" % (step, k): P[k].copy() for k in P})
        b = det_batch(step)
        loss, there, where, G = det_loss_and_grads(P, b["x"], b["has"], b["want"], pw)
        if step == 0:
            out.update({"G0__" + k: G[k] for k in P})
        out["loss"].append(loss)
        out["there"].append(there)
        out["where"].append(where)
        P, state = S.adam_update(P, G, state)
        if step == 0:
            out.update({"W1__" + k: P[k].copy() for k in P})
    out.update({"W%d__%s" % (DET_STEPS, k): P[k] for k in P})
    return {k: np.asarray(v) for k, v in out.items()}


def decode(out, keep=KEEP):
    """``decode`` of the script, for one scene's ``[5, 12, 12]``: ``(boxes [n, 4] as
    (cx, cy, w, h), scores [n], cells [n, 2])``."""
    p = _sig(out[0])
    r, c = np.nonzero(p > keep)
    xy = _sig(out[1:3, r, c])
    cx, cy = (c + xy[0]) * CELL, (r + xy[1]) * CELL
    s = 16 * np.exp(out[3:5, r, c])
    return np.stack([cx, cy, s[0], s[1]], 1).astype(np.float32), p[r, c], np.stack([r, c], 1)


def corners(boxes):
    """``box_convert(boxes, "cxcywh", "xyxy")``."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def iou(a, b):
    """``box_iou``: every box of ``a`` (corners) against every box of ``b``."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])   # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def nms(boxes, scores, threshold=NMS_IOU):
    """``torchvision.ops.nms``: keep the highest score, drop every box overlapping it by more
    than ``threshold``, repeat. Returns the kept indices, highest score first."""
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep = []
    o = iou(boxes, boxes)
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if o[i, j] <= threshold]
    return np.array(keep, int)


def find(P, canvas, keep=KEEP):
    """The detector on one scene, duplicates removed: ``dict(found [k, 4] corners, scores,
    before [n, 4] corners, before_scores)``."""
    boxes, sc, _cells = decode(det_forward(P, canvas[None, None])[0], keep)
    cor = corners(boxes)
    k = nms(cor, sc) if len(sc) else np.zeros(0, int)
    return dict(found=cor[k], scores=sc[k], before=cor, before_scores=sc)


def _match(found, scores, truth):
    """Greedy matching by score at IoU 0.5: per found box, True if it matches a face not
    already matched; and the IoU of each match."""
    real = corners(truth)
    ok, ious, used = [], [], set()
    if len(found) and len(real):
        o = iou(found, real)
        for q in np.argsort(-scores, kind="stable"):
            j = int(o[q].argmax())
            good = o[q, j] >= MATCH_IOU and j not in used
            if good:
                used.add(j)
                ious.append(o[q, j])
            ok.append((float(scores[q]), bool(good)))
    else:
        ok = [(float(s), False) for s in scores]
    return ok, ious


def det_eval(P):
    """Over the 200 test scenes: AP at IoU 0.5 (boxes kept from 0.05 up), the precision and
    recall curve, and at the script's 0.5: faces found, wrong boxes, boxes on not-faces,
    and the mean IoU of the right ones."""
    dets, n_true, ious = [], 0, []
    found05, wrong05, on_not = 0, 0, 0
    for sc in test_scenes():
        n_true += len(sc["boxes"])
        f = find(P, sc["canvas"], keep=0.05)
        ok, _io = _match(f["found"], f["scores"], sc["boxes"])
        dets += ok
        sure = f["scores"] >= KEEP
        ok5, io5 = _match(f["found"][sure], f["scores"][sure], sc["boxes"])
        found05 += sum(g for _s, g in ok5)
        wrong05 += sum(not g for _s, g in ok5)
        ious += io5
        nots = [(x + s / 2, y + s / 2, s, s) for x, y, s in sc["squares"][sc["faces"]:]]
        if len(nots) and sure.any():
            on_not += int((iou(f["found"][sure], corners(nots)) >= MATCH_IOU).any(1).sum())
    dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([g for _s, g in dets])
    fp = np.cumsum([not g for _s, g in dets])
    recall = tp / n_true
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(sum(p * (r - r0) for p, r, r0, (_s, g) in
                   zip(precision, recall, np.r_[0, recall[:-1]], dets, strict=True) if g))
    return dict(ap=ap, faces=n_true, found=int(found05), wrong=int(wrong05), on_not=on_not,
                mean_iou=float(np.mean(ious)) if ious else 0.0,
                recall=recall.astype(np.float32), precision=precision.astype(np.float32),
                scores=np.array([s for s, _g in dets], np.float32))


# ====================================================================== the recogniser
def rec_init(seed=0):
    return _init({"0": (8, 1, 3, 3), "3": (16, 8, 3, 3), "6": (32, 16, 3, 3),
                  "10": (EMBED, 512)}, seed, 4)


def rec_forward(P, x, keep_cache=False):
    """``encoder(x)`` (not yet scaled to length 1): ``[B, 32]``."""
    h, cache = _blocks(P, np.asarray(x, np.float32))
    flat = h.reshape(len(h), -1)
    e = (flat @ P["10.weight"].T + P["10.bias"]).astype(np.float32)
    return (e, (flat, h.shape, cache)) if keep_cache else e


def unit(e):
    return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def embed(P, x, chunk=100):
    return np.concatenate([unit(rec_forward(P, x[s:s + chunk])) for s in range(0, len(x), chunk)])


def triplets(step, seed=0, n=TRIPLETS):
    """``triplets`` of the script, from numpy's generator: photo numbers ``(i, j, k)``."""
    rng = np.random.default_rng([SEED, 5, int(seed), int(step)])
    who = rng.integers(0, TRAIN_PEOPLE, n)
    other = (who + rng.integers(1, TRAIN_PEOPLE, n)) % TRAIN_PEOPLE
    two = np.stack([rng.permutation(10)[:2] for _ in range(n)])
    return who * 10 + two[:, 0], who * 10 + two[:, 1], other * 10 + rng.integers(0, 10, n)


def triplet_loss_and_grads(P, i, j, k):
    """``F.triplet_margin_loss(a, p, n, margin=0.5)`` on the unit vectors of photos ``i``,
    ``j``, ``k`` and every parameter's gradient; also the per-triplet distances."""
    xs = small()
    outs = [rec_forward(P, xs[idx], keep_cache=True) for idx in (i, j, k)]
    es = [o[0] for o in outs]
    us = [unit(e) for e in es]
    a, p, n = us
    dap_v, dan_v = a - p + EPS, a - n + EPS
    dap, dan = np.linalg.norm(dap_v, axis=1), np.linalg.norm(dan_v, axis=1)
    per = np.maximum(dap - dan + MARGIN, 0)
    loss = float(per.mean())
    on = (per > 0).astype(np.float32) / len(per)
    g_ap = on[:, None] * dap_v / np.maximum(dap, 1e-12)[:, None]
    g_an = on[:, None] * dan_v / np.maximum(dan, 1e-12)[:, None]
    du = [g_ap - g_an, -g_ap, g_an]
    G = {key: np.zeros_like(v) for key, v in P.items()}
    for (e, (flat, hshape, cache)), u, d in zip(outs, us, du, strict=True):
        norm = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        de = ((d - u * (u * d).sum(1, keepdims=True)) / norm).astype(np.float32)
        G["10.weight"] += de.T @ flat
        G["10.bias"] += de.sum(0)
        dh = (de @ P["10.weight"]).reshape(hshape)
        for key, v in _blocks_back(P, dh, cache).items():
            G[key] += v
    return loss, {key: v.astype(np.float32) for key, v in G.items()}, dict(ap=dap, an=dan)


def _train_rec():
    P = rec_init()
    state = S.adam_start(P)
    out = dict(loss=[], active=[])
    for step in range(REC_STEPS):
        if step in REC_SNAPS:
            out.update({"W%d__%s" % (step, k): P[k].copy() for k in P})
        i, j, k = triplets(step)
        loss, G, d = triplet_loss_and_grads(P, i, j, k)
        if step == 0:
            out.update({"G0__" + key: G[key] for key in P})
        out["loss"].append(loss)
        out["active"].append(float((d["ap"] - d["an"] + MARGIN > 0).mean()))
        P, state = S.adam_update(P, G, state)
        if step == 0:
            out.update({"W1__" + key: P[key].copy() for key in P})
    out.update({"W%d__%s" % (REC_STEPS, k): P[k] for k in P})
    return {k: np.asarray(v) for k, v in out.items()}


def _pairs(v, who):
    i, j = np.triu_indices(len(v), 1)
    d = np.linalg.norm(v[i] - v[j] + 0.0, axis=1)
    return d, who[i] == who[j], i, j


def eer_threshold(d, same):
    """The threshold where the share of same-person pairs rejected equals the share of
    different-person pairs accepted (searched over the observed distances)."""
    best, at = 2.0, 0.0
    for t in np.unique(np.round(d, 4)):
        frr = (d[same] >= t).mean()
        far = (d[~same] < t).mean()
        gap = abs(frr - far)
        if gap < best:
            best, at = gap, float(t)
    return at


def rec_eval(v_all):
    """Verification and identification from the unit vectors of all 400 photos (``v_all``):
    the threshold chosen on the training people's pairs, then the test people's numbers."""
    who = photos()[1]
    tr, te = who < TRAIN_PEOPLE, who >= TRAIN_PEOPLE
    d_tr, s_tr, _i, _j = _pairs(v_all[tr], who[tr])
    t = eer_threshold(d_tr, s_tr)
    v, w = v_all[te], who[te]
    d, same, i, j = _pairs(v, w)
    acc_same = (d[same] < t).sum()
    acc_diff = (d[~same] < t).sum()
    out = dict(threshold=t, pairs=len(d), same_pairs=int(same.sum()),
               true_accept=int(acc_same), false_reject=int(same.sum() - acc_same),
               false_accept=int(acc_diff), true_reject=int((~same).sum() - acc_diff),
               d_same=d[same].astype(np.float32), d_diff=d[~same].astype(np.float32))
    out["right"] = out["true_accept"] + out["true_reject"]
    # the curve: false rejects against false accepts over thresholds
    ts = np.linspace(0, 2, 201)
    out["roc_t"] = ts
    out["frr"] = np.array([(d[same] >= x).mean() for x in ts])
    out["far"] = np.array([(d[~same] < x).mean() for x in ts])
    test_eer = eer_threshold(d, same)
    out["test_eer"] = float(max((d[same] >= test_eer).mean(), (d[~same] < test_eer).mean()))
    # identification: one photo each of people 30-34 enrolled, their other 45 photos probed
    gal = np.array([np.nonzero(who == p)[0][0] for p in ENROLLED])
    probes = np.array([q for p in ENROLLED for q in np.nonzero(who == p)[0][1:]])
    dd = np.linalg.norm(v_all[probes][:, None] - v_all[gal][None], axis=2)
    guess = np.array(ENROLLED)[dd.argmin(1)]
    truth = who[probes]
    out.update(identified=int((guess == truth).sum()), probes=len(probes),
               identified_sure=int(((guess == truth) & (dd.min(1) < t)).sum()),
               unsure=int((dd.min(1) >= t).sum()))
    strangers = np.array([q for p in STRANGERS for q in np.nonzero(who == p)[0]])
    ds = np.linalg.norm(v_all[strangers][:, None] - v_all[gal][None], axis=2).min(1)
    out.update(strangers=len(strangers), let_in=int((ds < t).sum()))
    # per person: the share of their own pairs rejected at t
    out["per_person_frr"] = np.array([((d[same] >= t) & (w[i][same] == p)).sum()
                                      / max(1, (w[i][same] == p).sum()) for p in range(30, 40)])
    out["per_person_far"] = np.array([((d[~same] < t) & ((w[i][~same] == p) | (w[j][~same] == p)))
                                      .sum() / max(1, ((w[i][~same] == p)
                                                       | (w[j][~same] == p)).sum())
                                      for p in range(30, 40)])
    return out


# ====================================================================== the runs
def _source(*fns):
    """A hash of the code a cached result depends on: these functions, or the whole module."""
    text = "".join(inspect.getsource(f) for f in fns) if fns else inspect.getsource(
        sys.modules[__name__])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


_DATA_CODE = (photos, small, _area, resize, background, scramble, places, scene, train_pool,
              test_pool, offsets, targets, _init, _pad1, _blocks, _blocks_back)
_DET_CODE = _DATA_CODE + (det_batch, det_init, det_forward, _sig, det_loss, det_loss_and_grads,
                          _train_det)


def _cache(name, build, code=(), **settings):
    settings.update(version=VERSION, source=_source(*code), data=olivetti.SHA256)
    return {k: np.asarray(v) for k, v in cached_npz(name, settings, build).items()}


@functools.lru_cache(maxsize=2)
def det_run(pw=PWS[0]):
    """A detector's training run (cached): per-step ``loss``, ``there``, ``where``; weights
    ``W<step>__<key>`` at :data:`DET_SNAPS`; ``G0__`` and ``W1__`` (the first step)."""
    return _cache("study-faces-det-%g" % pw, lambda: _train_det(pw), _DET_CODE, pw=pw,
                  steps=DET_STEPS, batch=BATCH, snaps=DET_SNAPS)


def det_params(step=DET_STEPS, pw=PWS[0]):
    if step == 0:
        return det_init()
    return {k: det_run(pw)["W%d__%s" % (step, k)] for k in DET_KEYS}


@functools.lru_cache(maxsize=1)
def rec_run():
    return _cache("study-faces-rec", _train_rec, _DATA_CODE + (
        rec_init, rec_forward, unit, triplets, triplet_loss_and_grads, _train_rec),
        steps=REC_STEPS, margin=MARGIN, snaps=REC_SNAPS)


def rec_params(step=REC_STEPS):
    if step == 0:
        return rec_init()
    return {k: rec_run()["W%d__%s" % (step, k)] for k in REC_KEYS}


@functools.lru_cache(maxsize=1)
def results():
    """Every test number (cached): the detectors' evaluations and the recogniser's, raw
    pixels' and the untrained network's."""
    def build():
        out = {}
        for pw in PWS:
            e = det_eval(det_params(DET_STEPS, pw))
            for k, v in e.items():
                out["det%g__%s" % (pw, k)] = v
        out.update({"pipe__" + k: v for k, v in _pipeline_rates_for_build().items()})
        for name, v in (("trained", embed(rec_params(), small())),
                        ("untrained", embed(rec_init(), small())),
                        ("pixels", unit(photos()[0].reshape(400, -1)))):
            for k, x in rec_eval(v).items():
                out["rec_%s__%s" % (name, k)] = x
        return out
    return _cache("study-faces-results", build, pws=PWS, scenes=TEST_SCENES,
                  det=[_source()])


def detection(pw=PWS[0]):
    p = "det%g__" % pw
    return {k[len(p):]: (v.item() if v.ndim == 0 else v) for k, v in results().items()
            if k.startswith(p)}


def recognition(name="trained"):
    p = "rec_%s__" % name
    return {k[len(p):]: (v.item() if v.ndim == 0 else v) for k, v in results().items()
            if k.startswith(p)}


def _identify(crop, v_gal):
    q = unit(rec_forward(rec_params(), crop[None, None]))
    d = np.linalg.norm(q - v_gal, axis=1)
    return int(ENROLLED[int(d.argmin())]), float(d.min())


@functools.lru_cache(maxsize=1)
def trained_threshold():
    """The verification threshold, chosen on the training people's pairs (as in rec_eval)."""
    who = photos()[1]
    tr = who < TRAIN_PEOPLE
    d, same, _i, _j = _pairs(embed(rec_params(), small())[tr], who[tr])
    return eer_threshold(d, same)


def _enrolled():
    who = photos()[1]
    gal = np.array([np.nonzero(who == p)[0][0] for p in ENROLLED])
    pool = [q for p in ENROLLED for q in np.nonzero(who == p)[0][1:]]
    return gal, pool


def pipeline_scene(faces=3):
    """The scene of "Find, cut out, compare": the first seeded scene of the enrolled people
    30-34 (never their enrolled photo) with ``faces`` faces, picked by that rule, not by how
    well it goes; what the detector finds and whom each box is matched to."""
    rng = np.random.default_rng([SEED, 6])
    gal, pool = _enrolled()
    while True:
        sc = scene(pool, rng)
        if sc["faces"] == faces:
            break
    f = find(det_params(), sc["canvas"])
    v_gal = embed(rec_params(), small())[gal]
    t = trained_threshold()
    who = photos()[1]
    truth = corners(sc["boxes"])
    named = []
    for box in np.round(f["found"]).astype(int):
        x1, y1, x2, y2 = (int(b) for b in np.clip(box, 0, SIDE))
        crop = resize_rect(sc["canvas"][y1:y2, x1:x2], 32)
        person, dist = _identify(crop, v_gal)
        o = iou(box[None], truth)[0] if len(truth) else np.zeros(0)
        k = int(o.argmax()) if len(o) and o.max() >= MATCH_IOU else None
        named.append(dict(box=[x1, y1, x2, y2], person=person, distance=dist,
                          sure=bool(dist < t), crop=crop, iou=float(o.max()) if len(o) else 0.0,
                          truth=None if k is None else int(who[sc["ids"][k]])))
    return dict(scene=sc, found=f, named=named,
                truth=[int(who[i]) for i in sc["ids"][:sc["faces"]]])


def _pipeline_rates(n=TEST_SCENES):
    """Over ``n`` seeded scenes of the enrolled people: of the faces, how many the detector
    finds (IoU 0.5), and how many are then named right: from the detector's box, from the
    true box (perfect detection), and from the true box when the name must also be sure."""
    rng = np.random.default_rng([SEED, 7])
    gal, pool = _enrolled()
    v_gal = embed(rec_params(), small())[gal]
    t = trained_threshold()
    who = photos()[1]
    out = dict(scenes=n, faces=0, boxes=0, found=0, right_detected=0, sure_right_detected=0,
               sure_wrong_detected=0, right_true=0, ious=[])
    for _ in range(n):
        sc = scene(pool, rng)
        f = find(det_params(), sc["canvas"])
        out["faces"] += sc["faces"]
        out["boxes"] += len(f["found"])
        truth = corners(sc["boxes"])
        for k, (x, y, s) in enumerate(sc["squares"][:sc["faces"]]):
            person, _d = _identify(resize_rect(sc["canvas"][y:y + s, x:x + s], 32), v_gal)
            out["right_true"] += person == who[sc["ids"][k]]
        for box in f["found"]:
            o = iou(box[None], truth)[0]
            if o.max() < MATCH_IOU:
                continue
            out["found"] += 1
            out["ious"].append(float(o.max()))
            b = np.clip(np.round(box), 0, SIDE).astype(int)
            person, d = _identify(resize_rect(sc["canvas"][b[1]:b[3], b[0]:b[2]], 32), v_gal)
            right = person == who[sc["ids"][int(o.argmax())]]
            out["right_detected"] += right
            out["sure_right_detected"] += right and d < t
            out["sure_wrong_detected"] += (not right) and d < t
    out["mean_iou"] = float(np.mean(out.pop("ious")))
    return {k: (int(v) if not isinstance(v, float) else v) for k, v in out.items()}


def _pipeline_rates_for_build():
    return _pipeline_rates()


def pipeline_rates():
    """:func:`_pipeline_rates`, cached with the other results."""
    r = results()
    return {k[len("pipe__"):]: r[k].item() for k in r if k.startswith("pipe__")}


def resize_rect(img, s):
    """``F.interpolate(img[None, None], size=s, mode="area")`` for a rectangle."""
    return (_area(s, img.shape[0]) @ img @ _area(s, img.shape[1]).T).astype(np.float32)


def summary():
    d, d1 = detection(10.0), detection(1.0)
    r, px, un = recognition(), recognition("pixels"), recognition("untrained")
    keep = ("ap", "faces", "found", "wrong", "on_not", "mean_iou")
    rk = ("threshold", "pairs", "same_pairs", "right", "false_accept", "false_reject",
          "test_eer", "identified", "identified_sure", "unsure", "probes", "let_in", "strangers")
    return dict(detector_params=count(det_init()), encoder_params=count(rec_init()),
                det={k: d[k] for k in keep}, det_unweighted={k: d1[k] for k in keep},
                rec={k: r[k] for k in rk}, pixels={k: px[k] for k in rk},
                untrained={k: un[k] for k in rk},
                det_loss=(float(det_run()["loss"][0]), float(det_run()["loss"][-100:].mean())),
                rec_loss=(float(rec_run()["loss"][0]), float(rec_run()["loss"][-100:].mean())))

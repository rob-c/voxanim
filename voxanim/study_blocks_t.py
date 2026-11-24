"""The numbers behind animations 238-240 of "Blocks and code" (group ``t``, the first three of
"Finding faces: bounding boxes"): a box, the grid of guesses, and the detector's layers.
Everything comes from :mod:`voxanim.study_faces`, the numpy twin of ``examples/faces.py``.

All three follow one scene: the first of the 200 held-out scenes (people 30-39 only,
:func:`~voxanim.study_faces.test_scenes`), a scene we made: two photos of faces pasted onto
textured paper at sizes 32 and 19, and two "not-faces" (photos cut into 8 x 8 blocks and
shuffled). Its boxes are exact because we pasted the faces. The face followed is its second
(:data:`FACE`): the first one's centre, (71, 48), lies exactly on the edge between two rows of
cells (48 = 6 x 8), a confusing first example.

* ``faces-a-box``: the box as the script's ``box = (x + s / 2, y + s / 2, s, s)`` writes it,
  (27.5, 61.5, 19, 19), and as corners after ``box_convert(..., "cxcywh", "xyxy")``, (18, 52,
  37, 71); then the box the trained detector draws for that face, and their IoU worked out
  as areas: overlap 308 over 361 + 383 - 308 = 436 together, 0.71 (``box_iou``). Right
  boxes overlap the truth by 0.78 on average over the 200 held-out scenes.
* ``faces-the-grid``: the scene under the 12 x 12 grid of 8 x 8-pixel cells; the face's
  centre falls in row 7, column 3 (``offsets``); the cell's numbers: 0.44 and 0.69 (where
  the centre is in the cell) and log(19 / 16) = 0.17 twice (the size); ``targets``: two
  cells of 144 have a face (``has`` 1), the other 142 say 0, the not-faces' cells too.
* ``faces-the-detector``: ``block`` three times and a 1 x 1 convolution, [1, 1, 96, 96] ->
  [1, 8, 48, 48] -> [1, 16, 24, 24] -> [1, 32, 12, 12] -> [1, 5, 12, 12], with the trained
  detector's real maps for the scene; the 1 x 1 convolution turns each cell's 32 numbers into
  5 (cell (7, 3): a face there with chance 0.97); 80 + 1,168 + 4,640 + 165 = 6,053 learned
  numbers; and what one cell can see: 3, 4, 8, 10, 18, then 22 x 22 pixels, checked on the
  trained model (changing every pixel outside the 22 x 22 square leaves the cell's 5
  numbers exactly as they were).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

from . import study_faces as F
from .blocks import code_lines, commas, shape, val
from .study import Panel, Study

SCENE = 0                       # test_scenes()[0]: the first held-out scene
FACE = 1                        # its second face (the first's centre is on a cell's edge)
SOURCE = ("the Olivetti faces (AT&T Laboratories Cambridge, 1992-94): the first of 200 "
          "held-out scenes of people 30-39, made as examples/faces.py makes them (photos "
          "pasted onto textured paper); voxanim.study_faces, the script's numpy twin")
MADE = ("the scene is made, not photographed: faces pasted onto paper, and not-faces that "
        "are photos shuffled in 8 x 8 blocks; that is why its boxes are exact")


# ====================================================================== words and numbers
def num(v, places=1):
    """A number as the script would write it, shortest: ``27.5``, ``19``, ``18.9``."""
    v = round(float(v), places)
    return ("%d" % v) if v == int(v) else ("%.*f" % (places, v)).rstrip("0").rstrip(".")


def two(v):
    return "%.2f" % float(v)


def tensor_text(a, indent=0):
    """``print(torch.tensor(a))`` for a 1-D or 2-D float tensor: integer-valued tensors as
    ``55.``, others with 4 decimals, the columns right-aligned to the widest."""
    a = np.atleast_1d(np.asarray(a, np.float64))
    whole = np.all(a == np.round(a))
    cells = [("%d." % v) if whole else ("%.4f" % v) for v in a.ravel()]
    cells = [c if c != "-0.0000" else "0.0000" for c in cells]
    w = max(len(c) for c in cells)
    cells = np.array([c.rjust(w) for c in cells]).reshape(a.shape)
    if a.ndim == 1:
        return "tensor([%s])" % ", ".join(cells)
    rows = ["[%s]" % ", ".join(r) for r in cells]
    return "tensor([" + (",\n" + " " * (8 + indent)).join(rows) + "])"


def area(box):
    """The area of a box given by its corners."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def overlap(a, b):
    """The corners of the overlap of two boxes (corners)."""
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


# ====================================================================== the scene
@functools.lru_cache(maxsize=1)
def the_scene():
    """The scene the three animations follow, with what they read off it."""
    sc = F.test_scenes()[SCENE]
    who = F.photos()[1]
    nf = sc["faces"]
    boxes = [tuple(float(v) for v in b) for b in sc["boxes"]]
    real = F.corners(boxes)
    return dict(canvas=sc["canvas"], boxes=boxes, squares=[tuple(int(v) for v in q)
                                                           for q in sc["squares"]],
                faces=nf, nots=len(sc["squares"]) - nf, real=real,
                people=[int(who[i]) for i in sc["ids"][:nf]],
                not_squares=[tuple(int(v) for v in q) for q in sc["squares"][nf:]])


def detector_on_scene():
    """The trained detector on the scene: its output ``[5, 12, 12]``, the maps after each
    block, and the boxes it keeps (``find``)."""
    s = the_scene()
    P = F.det_params()
    x = s["canvas"][None, None]
    out, (h, cache) = F.det_forward(P, x, keep_cache=True)
    maps = [cache[1][0][0], cache[2][0][0], h[0]]             # [8,48,48] [16,24,24] [32,12,12]
    return dict(out=out[0], maps=maps, found=F.find(P, s["canvas"]))


# ====================================================================== 238
def a_box():
    s = the_scene()
    x, y, side = s["squares"][FACE]
    box = s["boxes"][FACE]
    cx, cy, w, h = box
    if (cx, cy, w, h) != (x + side / 2, y + side / 2, side, side):
        raise ValueError("the box is not the pasted square's")
    cor = [float(v) for v in F.corners([box])[0]]
    if cor != [x, y, x + side, y + side]:
        raise ValueError("box_convert is not the pasted square's corners")
    det = detector_on_scene()["found"]
    found, scores = det["found"], det["scores"]
    o = F.iou(found, s["real"])                        # box_iou(found, real)
    j = int(np.argmax(o[:, FACE]))                     # the detector's box for this face
    guess = [float(v) for v in found[j]]
    inter = overlap(guess, cor)
    a_true, a_guess, a_inter = area(cor), area(guess), area(inter)
    union = a_true + a_guess - a_inter
    iou = a_inter / union
    if not np.isclose(iou, o[j, FACE]):
        raise ValueError("the worked IoU is not box_iou's")
    d = F.detection()
    paste = code_lines("fc-paste")
    meta = dict(
        source=SOURCE, canvas=s["canvas"].tolist(),
        evidence="exact arithmetic on the pasted boxes; the detector's box measured",
        made=MADE, scene=SCENE, face=FACE, faces=s["faces"], nots=s["nots"],
        side=F.SIDE, squares=s["squares"], boxes=[list(b) for b in s["boxes"]],
        paste=[x, y, side], box=list(box), centre_text="(%s, %s)" % (num(cx), num(cy)),
        parts_text=["cx = %s" % num(cx), "cy = %s" % num(cy), "w = %d" % side,
                    "h = %d" % side],
        box_text="(%s, %s, %d, %d)" % (num(cx), num(cy),
                                                                         side, side),
        half=side / 2, half_text=num(side / 2),
        corners=cor, corners_text="(%s)" % ", ".join(num(v) for v in cor),
        corner_sums=["%s = %s − %s = %s" % (n, num(c), num(side / 2), num(v)) for n, c, v in
                     (("x1", cx, cor[0]), ("y1", cy, cor[1]))] +
                    ["%s = %s + %s = %s" % (n, num(c), num(side / 2), num(v)) for n, c, v in
                     (("x2", cx, cor[2]), ("y2", cy, cor[3]))],
        real=s["real"].tolist(), printed_real=tensor_text(s["real"]),
        found=found.tolist(), found_scores=[float(v) for v in scores], guess_index=j,
        guess=guess, guess_text="(%s)" % ", ".join(num(v) for v in guess),
        guess_score=float(scores[j]), guess_score_text=two(scores[j]),
        inter=inter, area_true=a_true, area_guess=a_guess, area_inter=a_inter, union=union,
        iou=iou, iou_text=two(iou),
        area_texts=dict(true=commas(a_true), guess=commas(a_guess), inter=commas(a_inter),
                        union=commas(union)),
        union_sum="%s + %s − %s = %s" % (commas(a_true), commas(a_guess), commas(a_inter),
                                         commas(union)),
        iou_sum="%s ÷ %s = %s" % (commas(a_inter), commas(union), two(iou)),
        iou_matrix=o.tolist(), printed_iou=tensor_text(o),
        other_iou=float(o[1 - j, 1 - FACE]), other_iou_text=two(o[1 - j, 1 - FACE]),
        mean_iou=float(d["mean_iou"]), mean_iou_text=two(d["mean_iou"]),
        test_scenes=F.TEST_SCENES, paste_code=list(paste),
        limitations=[MADE, "the detector's box is the numpy twin's trained detector on this "
                     "scene; a torch run of the script trains from other random scenes and "
                     "draws a slightly different box",
                     "the face followed is the scene's second: the first one's centre lies "
                     "exactly on the edge between two rows of the grid (y = 48)"])
    canvas = s["canvas"].astype(float)
    panels = (
        Panel("The scene", "the first held-out scene: 2 faces pasted onto paper, 2 "
              "not-faces", "images", dict(values=canvas[None], labels=["held-out scene 1"])),
        Panel("One box, two ways", "(cx, cy, w, h) and box_convert's corners (x1, y1, x2, "
              "y2)", "bars", dict(values=np.array(list(box) + cor, float),
                                  labels=["cx", "cy", "w", "h", "x1", "y1", "x2", "y2"])),
        Panel("Two boxes for one face", "the true box and the trained detector's, as "
              "corners", "matrix", dict(values=np.array([cor, guess], float))),
        Panel("IoU as areas", "the overlap, both boxes together, and their ratio", "bars",
              dict(values=np.array([a_inter, union, iou], float),
                   labels=["overlap", "together", "IoU"])),
    )
    return Study("A box is four numbers", "A face's box as centre and size, or as two "
                 "corners, and how much two boxes agree", panels, meta)


# ====================================================================== 239
def the_grid():
    s = the_scene()
    box = s["boxes"][FACE]
    cx, cy, w, _h = box
    r, c, say = F.offsets(*box)
    has, want = F.targets(s["boxes"])
    if not (has[r, c] == 1 and np.allclose(want[:, r, c], say)):
        raise ValueError("targets does not hold the face's cell")
    cells = [[int(v) for v in F.offsets(*b)[:2]] for b in s["boxes"]]
    other = next(cl for cl in cells if cl != [r, c])
    nots = []
    for x, y, q in s["not_squares"]:
        rr, cc = int((y + q / 2) // F.CELL), int((x + q / 2) // F.CELL)
        nots.append([rr, cc, float(has[rr, cc])])
    lw = float(np.log(w / 16))
    say32 = np.float32(say)
    printed = "(%d, %d, %s)" % (r, c, tensor_text(say32))
    cut = printed.index(", 0.1719")
    wrapped = printed[:cut + 1] + "\n" + " " * len("(7, 3, tensor([") + printed[cut + 2:]
    meta = dict(
        source=SOURCE, canvas=s["canvas"].tolist(),
        evidence="exact arithmetic: offsets() and targets() on the pasted boxes",
        made=MADE, scene=SCENE, face=FACE, side=F.SIDE, squares=s["squares"],
        not_squares=s["not_squares"], cell=F.CELL, grid=F.GRID,
        grid_sum="%d ÷ %d = %d" % (F.SIDE, F.CELL, F.GRID), cells_total=F.GRID * F.GRID,
        box=list(box), centre_text="(%s, %s)" % (num(cx), num(cy)),
        row=r, col=c, row_div="%s ÷ %d = %s" % (num(cy), F.CELL, two(cy / F.CELL)),
        col_div="%s ÷ %d = %s" % (num(cx), F.CELL, two(cx / F.CELL)),
        cell_px=[c * F.CELL, r * F.CELL], dx=float(say[0]), dy=float(say[1]),
        dx_text=two(say[0]), dy_text=two(say[1]), in_cell=[float(cx - c * F.CELL),
                                                           float(cy - r * F.CELL)],
        size=w, lw=lw, lw_text=two(lw), log_sum="log(%s ÷ 16) = %s" % (num(w), two(lw)),
        size_range=[F.SIZES[0], F.SIZES[1]], size_logs=[0.0, float(np.log(2))],
        size_logs_text=[two(0.0), two(np.log(2))],
        say=[float(v) for v in say], five=[1.0] + [float(v) for v in say],
        five_text=["1"] + [two(v) for v in say],
        printed_offsets=printed, printed_offsets_wrapped=wrapped,
        face_cells=cells, other_cell=other, other_say=[float(v) for v in
                                                       want[:, other[0], other[1]]],
        has_ones=int(has.sum()), has_zeros=int(has.size - has.sum()), not_cells=nots,
        has=has.tolist(), want=want.tolist(),
        offsets_code=list(code_lines("fc-offsets")),
        targets_code=list(code_lines("fc-targets")),
        limitations=[MADE, "the other face's centre (71, 48) lies on the edge between rows "
                     "5 and 6; int(48 // 8) puts it in row 6"])
    panels = (
        Panel("The scene and its grid", "the held-out scene under 12 x 12 cells of 8 x 8 "
              "pixels", "images", dict(values=s["canvas"].astype(float)[None],
                                       labels=["held-out scene 1"])),
        Panel("has", "1 where a face's centre falls, 0 in the other cells", "matrix",
              dict(values=has.astype(float))),
        Panel("The face's cell", "what offsets() gives it: has, dx, dy, log(w/16), "
              "log(h/16)", "bars", dict(values=np.array([1.0] + list(say), float),
                                        labels=["has", "dx", "dy", "lw", "lh"])),
        Panel("Cells that say 1", "of the 144 cells", "bars",
              dict(values=np.array([has.sum(), has.size - has.sum()], float),
                   labels=["a face's centre", "nothing (0)"])),
    )
    return Study("A grid of guesses", "Each of 12 x 12 cells says whether a face is centred "
                 "in it, and where its box is", panels, meta)


# ====================================================================== 240
LAYERS = (("conv", 3), ("pool", 2), ("conv", 3), ("pool", 2), ("conv", 3), ("pool", 2),
          ("conv", 1))


def reach():
    """How far one unit sees, layer by layer: ``[(name, size, jump)]`` after each layer of
    the detector (size: the input pixels one unit depends on, across; jump: the input pixels
    between two neighbouring units)."""
    size, jump, out = 1, 1, []
    for name, k in LAYERS:
        size += (k - 1) * jump
        if name == "pool":
            jump *= k
        out.append((name, size, jump))
    return out


def field(r, c):
    """The input pixels output cell ``(r, c)`` depends on, as rows and columns ``[lo, hi]``
    (inclusive, before clipping to the scene): the cell's 8 x 8 pixels and 7 more each side."""
    grow = (reach()[-1][1] - F.CELL) // 2
    return ([r * F.CELL - grow, r * F.CELL + F.CELL - 1 + grow],
            [c * F.CELL - grow, c * F.CELL + F.CELL - 1 + grow])


def field_check(r, c, seed=0):
    """On the trained detector: every pixel outside cell ``(r, c)``'s field replaced by noise
    leaves its 5 numbers exactly as they were; replacing the field's border ring changes
    them. Returns ``(outside_change, ring_change)`` (largest absolute changes)."""
    s = the_scene()
    P = F.det_params()
    base = F.det_forward(P, s["canvas"][None, None])[0][:, r, c]
    (r0, r1), (c0, c1) = field(r, c)
    inside = np.zeros((F.SIDE, F.SIDE), bool)
    inside[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = True
    rng = np.random.default_rng(seed)
    noise = rng.random((F.SIDE, F.SIDE)).astype(np.float32)
    outside = np.where(inside, s["canvas"], noise)
    ring = inside.copy()
    ring[max(r0, 0) + 1:r1, max(c0, 0) + 1:c1] = False
    edge = np.where(ring, noise, s["canvas"])
    a = F.det_forward(P, outside[None, None])[0][:, r, c]
    b = F.det_forward(P, edge[None, None])[0][:, r, c]
    return float(np.abs(a - base).max()), float(np.abs(b - base).max())


def the_detector():
    s = the_scene()
    d = detector_on_scene()
    out, maps = d["out"], d["maps"]
    P = F.det_params()
    shapes = [[1, 1, F.SIDE, F.SIDE]] + [[1] + list(m.shape) for m in maps] + \
        [[1] + list(out.shape)]
    if [sh[1] for sh in shapes] != [1, 8, 16, 32, 5] or [sh[2] for sh in shapes] != \
            [96, 48, 24, 12, 12]:
        raise ValueError("the detector's shapes are not the script's")
    counts = [int(P[k + ".weight"].size + P[k + ".bias"].size) for k in ("0", "3", "6", "9")]
    total = F.count(P)
    if sum(counts) != total:
        raise ValueError("the layers' counts do not add up")
    kernels = [list(P[k + ".weight"].shape) for k in ("0", "3", "6", "9")]
    count_sums = ["%d × %d × %d + %d = %s" % (o, i, kh * kw, o, commas(n))
                  for (o, i, kh, kw), n in zip(kernels, counts, strict=True)]
    # the map drawn in front of each stack: the channel with the largest numbers on average
    front = [int(np.argmax(m.reshape(len(m), -1).mean(1))) for m in maps]
    sure = 1.0 / (1.0 + np.exp(-out[0].astype(np.float64)))
    r, c, _say = F.offsets(*s["boxes"][FACE])
    cells = [[int(v) for v in F.offsets(*b)[:2]] for b in s["boxes"]]
    above = [[int(a), int(b)] for a, b in np.argwhere(sure > F.KEEP)]
    steps = reach()
    lo_hi = field(r, c)
    outside_change, ring_change = field_check(r, c)
    if outside_change != 0.0 or ring_change <= 0.0:
        raise ValueError("cell (%d, %d) does not see exactly its 22 x 22 square" % (r, c))
    other = next(cl for cl in cells if cl != [r, c])
    other_side = s["boxes"][1 - FACE][2]
    meta = dict(
        source=SOURCE, canvas=s["canvas"].tolist(),
        evidence="measured: the trained detector on the held-out scene; the counts and the "
        "reach exact", made=MADE, scene=SCENE, face=FACE,
        shapes=shapes, shape_texts=[shape(sh) for sh in shapes],
        steps_text=[shape(a) + " → " + shape(b) for a, b in itertools.pairwise(shapes)],
        channels=[sh[1] for sh in shapes], sides=[sh[2] for sh in shapes],
        front=front, maps=[m[f].tolist() for m, f in zip(maps, front, strict=True)],
        map_max=[float(m.max()) for m in maps],
        out=out.tolist(), sure=sure.tolist(), face_cell=[r, c], face_cells=cells,
        cell_in=[float(v) for v in maps[2][:, r, c]], cell_out=[float(v) for v in out[:, r, c]],
        cell_out_text=[val(v) for v in out[:, r, c]],
        cell_sure=float(sure[r, c]), cell_sure_text=two(sure[r, c]),
        other_cell=other, other_cell_sure=float(sure[other[0], other[1]]),
        above=above, above_n=len(above),
        above_sure_text=[two(sure[a, b]) for a, b in above],
        counts=counts, count_texts=[commas(n) for n in counts], count_sums=count_sums,
        kernels=kernels, total=total, total_text=commas(total),
        one_by_one="5 × 32 + 5 = %d" % counts[3],
        reach=[[n, sz, j] for n, sz, j in steps], reach_sizes=[sz for _n, sz, _j in steps],
        field=lo_hi, field_side=steps[-1][1], outside_change=outside_change,
        ring_change=ring_change, other_face_side=other_side,
        block_code=list(code_lines("fc-block")), detector_code=list(code_lines("fc-detector")),
        limitations=[MADE, "the maps are the numpy twin's trained detector on one held-out "
                     "scene; each stack shows in front the channel whose numbers are "
                     "largest on average",
                     "the bigger face's centre lies on the edge between rows 5 and 6, and "
                     "the detector's surest cells for it are in row 5, one row above its "
                     "target cell: %d cells say a face with chance over 0.5" % len(above),
                     "what a cell can see is a 22 x 22 square, but max-pooling means only "
                     "some of those pixels matter for a given scene"])
    panels = (
        Panel("The maps", "the front map of each stack: after block 1, 2 and 3", "images",
              dict(values=np.stack([np.kron(m[f], np.ones((96 // len(m[f]),) * 2))
                                    for m, f in zip(maps, front, strict=True)]),
                   labels=["block %d" % (i + 1) for i in range(3)])),
        Panel("A face here?", "the first of the 5 numbers through a sigmoid, per cell",
              "matrix", dict(values=sure)),
        Panel("Learned numbers", "each layer's weights and biases", "bars",
              dict(values=np.array(counts, float),
                   labels=["block(1, 8)", "block(8, 16)", "block(16, 32)", "1 x 1 conv"])),
        Panel("What one cell sees", "the pixels across one unit depends on, layer by layer",
              "curve", dict(x=np.arange(1, len(steps) + 1, dtype=float),
                            ys=np.array([[sz for _n, sz, _j in steps]], float),
                            labels=["pixels across"])),
    )
    return Study("From pixels to a grid", "Three blocks take a 96 x 96 scene to a 12 x 12 "
                 "grid; a 1 x 1 convolution gives each cell 5 numbers", panels, meta)


STUDIES = {"faces-a-box": a_box, "faces-the-grid": the_grid, "faces-the-detector": the_detector}

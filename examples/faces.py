"""Finding faces and telling them apart: the program of the face animations of "Blocks and code".

    python examples/faces.py

Two small CNNs trained on the Olivetti faces (AT&T Laboratories Cambridge, 1992-94: ten photos
of each of 40 people, taken for face-recognition research; scikit-learn downloads them, 4 MB).
People 0-29 train both models; people 30-39 are kept for testing, so every test is on faces
the models never saw.

1. A detector. Scenes of 96 x 96: one to three faces pasted in at sizes 16 to 32, and up to
   two "not-faces" (a photo cut into 8 x 8 blocks and shuffled: the same pixels, no face).
   It answers on a grid of 12 x 12 cells: is a face's centre in this cell, and where is its
   box. Then the boxes it is sure of, duplicates removed (non-maximum suppression).
2. A recogniser. A face in, 32 numbers out, trained with triplets so that two photos of one
   person land closer than photos of two people. Then verification (the same person?),
   identification (which of the people enrolled?), strangers, and the two models together.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen
(voxanim.blocks.code_lines("NAME")), and lines of code are kept to 41 characters.
"""
# region: fc-imports
import math

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces
from torch import nn
from torchvision.ops import box_convert, box_iou, nms

# endregion

torch.manual_seed(0)
# region: fc-data
faces = fetch_olivetti_faces()            # 400 photos, 40 people
photos = torch.tensor(faces.images)       # [400, 64, 64], 0..1
person = torch.tensor(faces.target)       # photo i: person i // 10
train = person < 30                       # people 0-29 train
test = ~train                             # people 30-39: unseen
# endregion

SIDE, CELL, GRID = 96, 8, 12               # a scene, a cell, cells across
FMT = ("cxcywh", "xyxy")                   # centre and size -> corners
THRESHOLD = None                           # "the same person" below this: set on people 0-29


def background():
    """Grey paper with a soft texture: noise averaged over 9 x 9."""
    noise = torch.rand(1, 1, SIDE + 8, SIDE + 8)
    return F.avg_pool2d(noise, 9, 1)[0]                 # [1, 96, 96]


def scramble(img):
    """A not-face: the photo cut into 8 x 8 blocks, the blocks shuffled."""
    b = img.reshape(8, 8, 8, 8).permute(0, 2, 1, 3).reshape(64, 8, 8)
    b = b[torch.randperm(64)]
    return b.reshape(8, 8, 8, 8).permute(0, 2, 1, 3).reshape(64, 64)


def places(n, lo=16, hi=32):
    """n squares (x, y, s) inside the scene that do not overlap."""
    out = []
    while len(out) < n:
        s = int(torch.randint(lo, hi + 1, ()))
        x, y = (int(v) for v in torch.randint(0, SIDE - s + 1, (2,)))
        if all(x + s <= a or a + t <= x or y + s <= b or b + t <= y for a, b, t in out):
            out.append((x, y, s))
    return out


def scene(pool):
    """One scene: 1-3 faces from the photos in ``pool`` and 0-2 not-faces. Returns the
    picture and the faces' boxes as (centre x, centre y, width, height)."""
    n_faces, n_not = int(torch.randint(1, 4, ())), int(torch.randint(0, 3, ()))
    canvas, boxes = background(), []
    ids = pool[torch.randint(len(pool), (n_faces + n_not,))]
    for k, (x, y, s) in enumerate(places(n_faces + n_not)):
        img = photos[ids[k]] if k < n_faces else scramble(photos[ids[k]])
        # region: fc-paste
        face = F.interpolate(img[None, None],
                             size=s, mode="area")  # 64 x 64 -> s x s
        canvas[0, y:y + s, x:x + s] = face[0, 0]
        box = (x + s / 2, y + s / 2, s, s)  # centre x, centre y, w, h
        # endregion
        if k < n_faces:
            boxes.append(box)
    return canvas, boxes


# region: fc-offsets
def offsets(x, y, w, h):                  # one face's box
    r, c = int(y // CELL), int(x // CELL)  # the cell its centre is in
    dx, dy = x / CELL - c, y / CELL - r   # the centre, in that cell
    lw = math.log(w / 16)                 # the size, as log(w / 16)
    lh = math.log(h / 16)
    say = torch.tensor([dx, dy, lw, lh])
    return r, c, say
# endregion


# region: fc-targets
def targets(boxes):                       # (centre x, centre y, w, h)
    has = torch.zeros(GRID, GRID)         # 1: a face's centre is here
    want = torch.zeros(4, GRID, GRID)     # and its box, from this cell
    for box in boxes:
        r, c, say = offsets(*box)         # its cell, and what it says
        has[r, c] = 1
        want[:, r, c] = say
    return has, want
# endregion


# region: fc-block
def block(i, o):                          # look closer, then halve
    conv = nn.Conv2d(i, o, 3, padding=1)  # o stamps of i x 3 x 3
    pool = nn.MaxPool2d(2)                # then half as wide
    return [conv, nn.ReLU(), pool]
# endregion


# region: fc-detector
detector = nn.Sequential(
    *block(1, 8), *block(8, 16),          # 96 -> 48 -> 24
    *block(16, 32),                       # -> 12: a grid of 12 x 12
    nn.Conv2d(32, 5, 1),                  # 5 numbers in every cell
)
# endregion

# region: fc-weight
PW = torch.tensor(10.0)                   # a face's cell counts 10 times
# endregion


def detector_loss(out, has, want, pw=PW):
    # region: fc-loss
    obj, box = out[:, 0], out[:, 1:]      # a face here? where?
    bce = F.binary_cross_entropy_with_logits
    there = bce(obj, has, pos_weight=pw)  # every cell
    xy, wh = box[:, :2].sigmoid(), box[:, 2:]
    miss = ((xy - want[:, :2]) ** 2).sum(1)  # the centre
    miss += ((wh - want[:, 2:]) ** 2).sum(1)  # and the size
    where = (miss * has).sum() / has.sum()  # face cells only
    loss = there + 5 * where              # where counts 5 times
    # endregion
    return loss


def train_detector(steps=6000, batch=16, pw=PW):
    optimizer = torch.optim.Adam(detector.parameters())
    pool = torch.nonzero(train)[:, 0]
    for step in range(steps):
        scenes = [scene(pool) for _ in range(batch)]
        x = torch.stack([c for c, _ in scenes])
        has, want = (torch.stack(t) for t in zip(*(targets(b) for _, b in scenes),
                                                     strict=True))
        # region: fc-train
        out = detector(x)                 # [16, 5, 12, 12]
        loss = detector_loss(out, has, want, pw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # endregion
        if step % 1000 == 0:
            print("detector", step, loss.item())


def decode(out, keep=0.5):
    # region: fc-decode
    p = out[0].sigmoid()                  # [12, 12]: how sure, per cell
    r, c = torch.nonzero(p > keep).T      # the cells that say "face"
    xy = out[1:3, r, c].sigmoid()
    cx = (c + xy[0]) * CELL               # back to pixels
    cy = (r + xy[1]) * CELL
    s = 16 * out[3:5, r, c].exp()         # width and height
    boxes = torch.stack([cx, cy, *s], 1)
    # endregion
    return boxes, p[r, c]


def find(canvas):
    """The detector's boxes on one scene, duplicates removed: corners and scores."""
    with torch.no_grad():
        boxes, score = decode(detector(canvas[None])[0])
    # region: fc-nms
    corners = box_convert(boxes, *FMT)    # corners: x1, y1, x2, y2
    kept = nms(corners, score, 0.3)       # drop boxes that overlap
    found = corners[kept]                 # one box per face, we hope
    # endregion
    return found, score[kept]


def show_detector():
    canvas, truth = scene(torch.nonzero(test)[:, 0])
    found, _score = find(canvas)
    # region: fc-iou
    real = torch.tensor(truth)
    real = box_convert(real, *FMT)        # corners, as found
    print(box_iou(found, real))           # 1: exactly the same box
    # endregion


# ---------------------------------------------------------------- telling faces apart
small = F.avg_pool2d(photos[:, None], 2)  # [400, 1, 32, 32]

# region: fc-encoder
encoder = nn.Sequential(
    *block(1, 8), *block(8, 16),          # 32 -> 16 -> 8
    *block(16, 32),                       # -> 4 x 4
    nn.Flatten(),                         # 32 x 4 x 4 = 512
    nn.Linear(512, 32),                   # a face: 32 numbers
)
# endregion


def triplets(n=32):
    """n (anchor, positive, negative) photo numbers from the training people."""
    who = torch.randint(30, (n,))
    other = (who + torch.randint(1, 30, (n,))) % 30
    two = torch.stack([torch.randperm(10)[:2] for _ in range(n)])
    return who * 10 + two[:, 0], who * 10 + two[:, 1], other * 10 + torch.randint(10, (n,))


def train_encoder(steps=1500):
    optimizer = torch.optim.Adam(encoder.parameters())
    for step in range(steps):
        i, j, k = triplets()
        anchor, same, other = small[i], small[j], small[k]
        # region: fc-triplet
        a = F.normalize(encoder(anchor))  # a face: 32 numbers, length 1
        p = F.normalize(encoder(same))    # that person, another photo
        n = F.normalize(encoder(other))   # somebody else
        loss = F.triplet_margin_loss(
            a, p, n, margin=0.5)          # n 0.5 further than p
        # endregion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print("encoder", step, loss.item())


def choose_threshold():
    """The distance where, over the training people's pairs, the share of same-person pairs
    rejected equals the share of different-person pairs accepted."""
    global THRESHOLD
    with torch.no_grad():
        v = F.normalize(encoder(small[train]))
    i, j = torch.triu_indices(len(v), len(v), 1)
    d = (v[i] - v[j]).norm(dim=1)
    same = person[train][i] == person[train][j]
    ts = torch.unique(d.round(decimals=4))
    gap = [abs((d[same] >= t).float().mean() - (d[~same] < t).float().mean()) for t in ts]
    THRESHOLD = float(ts[int(torch.tensor(gap).argmin())])
    print("threshold", THRESHOLD)


def show_recogniser():
    with torch.no_grad():
        # region: fc-verify
        v = F.normalize(encoder(small[test]))  # 100 unseen photos
        i, j = torch.triu_indices(100, 100, 1)  # all 4,950 pairs
        d = (v[i] - v[j]).norm(dim=1)     # how far apart, each pair
        same = person[test][i] == person[test][j]
        yes = d < THRESHOLD               # "the same person"
        print("right:", (yes == same).sum())
        # endregion

        # region: fc-identify
        gallery = v[:50:10]               # people 30-34: enrolled
        others = torch.arange(50) % 10 > 0  # their 9 other photos
        probes = v[:50][others]
        d = torch.cdist(probes, gallery)  # [45, 5]
        who = d.argmin(1)                 # the nearest enrolled face
        # endregion
        print("identified:", (who == torch.arange(45) // 9).sum(), "of 45")

        # region: fc-strangers
        strangers = v[50:]                # people 35-39: not enrolled
        near = torch.cdist(strangers, gallery)
        d = near.min(1).values            # the nearest enrolled face
        print("let in:", (d < THRESHOLD).sum())  # of 50
        # endregion


def show_pipeline():
    enrolled = torch.tensor([i for i in range(300, 350) if i % 10])  # not the gallery's photos
    canvas, _truth = scene(enrolled)
    found, _score = find(canvas)
    with torch.no_grad():
        gallery = F.normalize(encoder(small[300:350:10]))
        for box in found.round().int():
            # region: fc-pipeline
            x1, y1, x2, y2 = box.tolist()
            crop = canvas[:, y1:y2, x1:x2]    # the face, cut out
            crop = F.interpolate(crop[None], size=32,
                                 mode="area")  # as big as in training
            q = F.normalize(encoder(crop))    # 32 numbers
            d = torch.cdist(q, gallery)[0]
            who = int(d.argmin()) + 30        # the nearest enrolled
            # endregion
            print(box.tolist(), "person", who, "sure" if d.min() < THRESHOLD else "unsure")


if __name__ == "__main__":
    train_detector()
    show_detector()
    train_encoder()
    choose_threshold()
    show_recogniser()
    show_pipeline()

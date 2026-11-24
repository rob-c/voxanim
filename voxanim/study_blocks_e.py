"""The numbers behind animations 197-199 of the "Blocks and code" series (group ``e``):
logits and softmax, a recap of every block so far, and the data pipeline.

* ``blocks-logits-and-softmax``: the course's trained model (:func:`voxanim.study_course.params`
  after the SGD epoch) on one held-out digit: its 10 logits, their softmax (bars that add up
  to 1), the loss ``nn.CrossEntropyLoss`` gives from the logits (``-ln`` of the right
  digit's share), and what goes wrong with a softmax in front of it: the shares squashed
  flat, so the loss can never fall below ``ln(1 + 9/e) = 1.46``; the real run trained that
  way (``study_course`` run ``double_softmax``) and its held-out accuracy.
* ``blocks-which-is-which``: the eleven blocks of 185-197, each with its line of code (from
  the regions of ``examples/mnist_blocks.py`` and ``examples/blocks_parts.py``), its input
  and output shapes and its weight count, all worked out here from the settings in those
  lines (parsed from the code, not typed in) by the rules PyTorch follows; the test checks
  them against torch's own modules when torch is installed.
* ``blocks-the-data-pipeline``: ``MNIST`` gives 60,000 digits of pixels 0..255;
  ``ToTensor`` divides by 255; ``Normalize`` subtracts the mean and divides by the spread of
  every training pixel (:func:`voxanim.study_course.pixel_stats`: 0.13066 and 0.30811, the
  script rounds them to 0.1307 and 0.3081); a shuffling ``DataLoader`` cuts the shuffled
  order into 938 batches of 64 (the last of 32), a new order every epoch.
"""

from __future__ import annotations

import ast
import math
import re

import numpy as np

from . import study_course as course
from .blocks import PARTS, code_line, code_lines, shape, torch_size
from .study import Panel, Study
from .study_blocks_a import zoom_window

SIDE = course.SIDE
SOURCE = ("the course's model after one epoch of SGD (voxanim.study_course, as "
          "examples/mnist_blocks.py trains it), on the MNIST test digits")
SURE = 0.9                  # 197: the first test digit the model gets right, but below this


# ====================================================================== 197
def softmax(z):
    """``z.softmax(dim=-1)`` in float64."""
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def double_softmax_floor(classes=course.CLASSES):
    """The lowest loss a model ending in ``nn.Softmax`` can reach under
    ``nn.CrossEntropyLoss``: its output ``p`` goes through softmax again, so the right
    digit's share is at most ``e / (e + classes - 1)`` (``p`` one-hot): the loss is at least
    ``ln(1 + (classes - 1) / e)``."""
    return math.log(1.0 + (classes - 1) / math.e)


def chosen_digit(P=None, sure=SURE):
    """The first test digit the trained model gets right with less than ``sure`` of its
    probability on the right digit (so that the other bars are not all invisible)."""
    P = course.params(course.EPOCH_STEPS) if P is None else P
    _x, _y, xt, yt = course.data()
    for s in range(0, len(xt), 500):
        p = softmax(course.forward(P, xt[s:s + 500])["logits"])
        for i in range(len(p)):
            if p[i].argmax() == yt[s + i] and p[i, yt[s + i]] < sure:
                return s + i
    raise ValueError("no such digit")


def logits_and_softmax():
    P = course.params(course.EPOCH_STEPS)
    i = chosen_digit(P)
    _x, _y, xt, yt = course.data()
    x, y = xt[i], int(yt[i])
    logits32 = course.forward(P, x[None])["logits"][0]
    z = logits32.astype(np.float64)
    p = softmax(z)
    loss = float(-np.log(p[y]))
    torch_loss = float(course.loss_each(logits32[None], [y])[0])     # CrossEntropyLoss's own
    if abs(loss - torch_loss) > 1e-5 or abs(p.sum() - 1) > 1e-12:
        raise ValueError("softmax or the loss does not add up")
    q = softmax(p)                                  # the mistake: softmax, then the loss's own
    loss_twice = float(-np.log(q[y]))
    floor = double_softmax_floor()
    top = math.e / (math.e + course.CLASSES - 1)
    order = [int(d) for d in np.argsort(-p, kind="stable")]
    second = order[1]
    cum = np.cumsum(p)                              # adding the bars up, digit 0 to 9
    b = course.first_batch()
    batch_logits = course.forward(P, b["x"])["logits"]
    s = course.summary()
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the trained model's logits",
        index=int(i), label=y, sure=SURE, logits=[float(v) for v in z],
        probs=[float(v) for v in p], cumulative=[float(v) for v in cum],
        prediction=int(np.argmax(z)), second=second, order=order,
        prob_right=float(p[y]), prob_second=float(p[second]),
        logit_right=float(z[y]), loss=loss, loss_check=torch_loss,
        twice=[float(v) for v in q], twice_right=float(q[y]), loss_twice=loss_twice,
        floor=floor, floor_text="ln(1 + 9/e) = %.2f" % floor, top_twice=top,
        shape=[1, course.CLASSES], batch_shape=list(batch_logits.shape),
        batch_shape_text=shape(batch_logits.shape),
        accuracy=s["sgd"]["accuracy"], accuracy_twice=s["double_softmax"]["accuracy"],
        test_loss=s["sgd"]["test_loss"], test_loss_twice=s["double_softmax"]["test_loss"],
        model=list(code_lines("model")), step=list(code_lines("step")),
        softmax_line=code_line("softmax", ".softmax("), loss_line=code_line("loss", "Loss"),
        mistake=list(code_lines("mistake-double-softmax")),
        limitations=["the digit is chosen: the first test digit the model gets right with "
                     "less than %d%% on it, so that the other bars show; most test digits "
                     "get over 99%%" % round(100 * SURE),
                     "the double-softmax accuracy is the numpy twin's run of the script "
                     "with nn.Softmax(dim=1) added (voxanim.study_course)"])
    digits = [str(d) for d in range(course.CLASSES)]
    panels = (
        Panel("The 10 logits", "model(x) for test digit %d (a %d): one score per digit" % (i, y),
              "bars", dict(values=z, labels=digits)),
        Panel("Softmax", "logits.softmax(dim=1): positive, and adding up to 1", "bars",
              dict(values=p, labels=digits)),
        Panel("Softmax twice", "what the loss sees if the model ends in nn.Softmax: the "
              "shares squashed between %.2f and %.2f" % (q.min(), q.max()), "bars",
              dict(values=q, labels=digits)),
        Panel("The digit", "test digit %d, label %d" % (i, y), "images",
              dict(values=course.pixels()[1][i].reshape(1, SIDE, SIDE).astype(float),
                   labels=["label %d" % y])),
    )
    return Study("Logits and softmax", "The model's 10 scores, softmax turning them into "
                 "shares that add up to 1, and the loss that takes the scores", panels, meta)


# ====================================================================== 198
# The blocks of 185-197 in the order they came: (name, region, text in the line, input)
# where the input is the name of the tensor the block is given (see INPUTS)
RECAP = (
    ("Flatten", "model", "Flatten", "batch"),
    ("Linear", "model", "Linear(784, 64)", "flat"),
    ("ReLU", "model", "ReLU", "hidden"),
    ("Conv2d", "conv", "Conv2d", "x"),
    ("MaxPool2d", "pool", "MaxPool2d", "maps"),
    ("Dropout", "dropout", "Dropout(", "h"),
    ("BatchNorm1d", "batchnorm", "BatchNorm1d", "h"),
    ("LayerNorm", "layernorm", "LayerNorm", "h"),
    ("Embedding", "embedding", "Embedding", "y"),
    ("MultiheadAttention", "attention", "MultiheadAttention", "q"),
    ("softmax", "softmax", ".softmax(", "logits"),
)
LEARNS = ("Linear", "Conv2d", "BatchNorm1d", "LayerNorm", "Embedding", "MultiheadAttention")


def made_up_inputs(path=PARTS):
    """The shapes of the tensors ``examples/blocks_parts.py`` makes up to run its blocks on
    (``x = torch.randn(64, 1, 28, 28)`` ...): ``{name: [dims]}``, read from the file."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        f = node.value.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "torch" and f.attr in ("randn", "randint")):
            continue
        args = [ast.literal_eval(a) for a in node.value.args]
        dims = list(args[2]) if f.attr == "randint" else args
        for t in node.targets:
            names = [t] if isinstance(t, ast.Name) else []
            for n in names:
                out.setdefault(n.id, [int(d) for d in dims])
    return out


def block_call(region, contains):
    """The call that makes a block, from its region's lines: ``(text, name, args,
    kwargs)``, e.g. ``("nn.Conv2d(1, 4, 3)", "Conv2d", [1, 4, 3], {})``. A call spread
    over two lines (``nn.MultiheadAttention(`` ...) is joined; ``logits.softmax(dim=1)``
    is a method, not a module."""
    lines = code_lines(region)
    at = next(i for i, s in enumerate(lines) if contains in s)
    src = lines[at].strip()
    k = at
    while src.count("(") > src.count(")"):
        k += 1
        src += lines[k].strip()
    src = src.rstrip(",")
    m = re.search(r"((?:nn\.\w+|\w+\.softmax)\(.*\))$", src)
    if m is None:
        raise ValueError("no block call in %r" % src)
    text = m.group(1)
    call = ast.parse(text, mode="eval").body
    name = call.func.attr
    args = [ast.literal_eval(a) for a in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return text, name, args, kwargs


def output_shape(name, args, kwargs, dims):
    """The shape a block gives for an input of shape ``dims``, by PyTorch's rules (the
    defaults: no padding, stride 1 for a convolution, stride = size for pooling)."""
    dims = [int(d) for d in dims]
    if name == "Flatten":
        return [dims[0], int(np.prod(dims[1:]))]
    if name == "Linear":
        if dims[-1] != args[0]:
            raise ValueError("Linear(%d, ...) given %s" % (args[0], dims))
        return dims[:-1] + [args[1]]
    if name == "Conv2d":
        cin, cout, k = args[:3]
        if dims[1] != cin:
            raise ValueError("Conv2d(%d, ...) given %s" % (cin, dims))
        return [dims[0], cout, dims[2] - k + 1, dims[3] - k + 1]
    if name == "MaxPool2d":
        k = args[0]
        return dims[:2] + [dims[2] // k, dims[3] // k]
    if name == "Embedding":
        return dims + [args[1]]
    if name in ("BatchNorm1d", "LayerNorm", "MultiheadAttention") and dims[-1] != args[0]:
        raise ValueError("%s(%d, ...) given %s" % (name, args[0], dims))
    return list(dims)          # ReLU, Dropout, the norms, attention, softmax: the same shape


def weights(name, args, kwargs):
    """``(count, sum)``: the numbers a block learns (``sum(p.numel() for p in
    block.parameters())``) and how they add up, by PyTorch's defaults (a bias everywhere,
    ``elementwise_affine`` and ``affine`` on)."""
    if name == "Linear":
        a, b = args[:2]
        return a * b + b, "%d × %d + %d" % (a, b, b)
    if name == "Conv2d":
        cin, cout, k = args[:3]
        return cout * cin * k * k + cout, "%d × %d × %d × %d + %d" % (cout, cin, k, k, cout)
    if name in ("BatchNorm1d", "LayerNorm"):
        n = args[0]
        return 2 * n, "%d + %d" % (n, n)
    if name == "Embedding":
        n, d = args[:2]
        return n * d, "%d × %d" % (n, d)
    if name == "MultiheadAttention":
        e = args[0]               # q, k, v and out: four e x e grids, four bias strips of e
        return 4 * e * e + 4 * e, "4 × %d × %d + 4 × %d" % (e, e, e)
    return 0, "0"


def which_is_which():
    made = made_up_inputs()
    batch = [course.BATCH, 1, SIDE, SIDE]
    shapes = dict(batch=batch)
    blocks = []
    for name, region, contains, given in RECAP:
        text, kind, args, kwargs = block_call(region, contains)
        if kind != name:
            raise ValueError("region %r makes %s, not %s" % (region, kind, name))
        dims = shapes.get(given, made.get(given))
        if dims is None:
            raise ValueError("no shape for %r" % given)
        out = output_shape(kind, args, kwargs, dims)
        count, how = weights(kind, args, kwargs)
        line = next(s for s in code_lines(region) if contains in s).strip()
        blocks.append(dict(name=name, region=region, line=line, call=text, args=args,
                           kwargs=kwargs, given=given, shape_in=dims, shape_out=out,
                           shape_in_text=shape(dims), shape_out_text=shape(out),
                           weights=count, how=how, learns=count > 0))
        # what the next block of the same program is given
        follow = {"Flatten": "flat", "Linear": "hidden", "Conv2d": "maps"}
        if name in follow:
            shapes[follow[name]] = out
    if [b["name"] for b in blocks if b["learns"]] != list(LEARNS):
        raise ValueError("the blocks that learn are not the expected ones")
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    meta = dict(
        source="the settings in the code of examples/mnist_blocks.py and "
               "examples/blocks_parts.py", evidence="exact arithmetic from each block's settings",
        blocks=blocks, groups=groups, made_up=made, total=len(blocks),
        learning=sum(b["learns"] for b in blocks),
        limitations=["the inputs of the part-2 blocks are the made-up tensors of "
                     "examples/blocks_parts.py, shaped like an MNIST batch; the shapes and "
                     "counts follow PyTorch's rules and defaults (checked against torch by "
                     "the tests when it is installed)",
                     "BatchNorm1d also keeps 2 × 64 running averages, which it does not "
                     "learn: they are not counted"])
    names = [b["name"] for b in blocks]
    panels = (
        Panel("Weights", "the numbers each block learns", "bars",
              dict(values=np.array([b["weights"] for b in blocks], float), labels=names)),
        Panel("Numbers in", "how many numbers each block is given", "bars",
              dict(values=np.array([np.prod(b["shape_in"]) for b in blocks], float),
                   labels=names)),
        Panel("Numbers out", "how many numbers each block gives", "bars",
              dict(values=np.array([np.prod(b["shape_out"]) for b in blocks], float),
                   labels=names)),
        Panel("Learns or not", "1: the block has weights; 0: it only reshapes or rescales",
              "bars", dict(values=np.array([b["learns"] for b in blocks], float),
                           labels=names)),
    )
    return Study("Which block is which", "Every block so far, with its line of code, its "
                 "shapes and the numbers it learns", panels, meta)


# ====================================================================== 199
DIGIT = 0                     # train_set[0], the digit followed through the pipeline
PATCH = 3                     # its window read out: PATCH x PATCH
SHOWN = 8                     # indices of the shuffled order read out
BELT = 3                      # 199: the first batches on the conveyor belt, then the last


def data_pipeline():
    raw01 = course.pixels()[0]                          # ToTensor's output, 0..1
    _x, y, _xt, _yt = course.data()
    img01 = raw01[DIGIT].reshape(SIDE, SIDE)
    img255 = np.round(img01.astype(np.float64) * 255).astype(int)
    if not np.allclose(img255 / 255.0, img01, atol=1e-7):
        raise ValueError("the pixels are not whole numbers over 255")
    norm = course.digit(DIGIT)[0][0]
    stats = course.pixel_stats()
    r0, c0 = zoom_window(norm, PATCH)
    sl = np.s_[r0:r0 + PATCH, c0:c0 + PATCH]
    # one pixel followed through: the grey one nearest half way between paper and ink
    win = img255[sl]
    v, (pi, pj) = min((abs(int(v) - 128), (int(v), (i, j)))
                      for (i, j), v in np.ndenumerate(win))[1]
    p01 = v / 255.0
    pn = (np.float32(p01) - np.float32(course.MEAN)) / np.float32(course.STD)
    if not np.isclose(pn, norm[r0 + pi, c0 + pj], atol=1e-6):
        raise ValueError("Normalize does not give the data's value")
    hist, edges = np.histogram(raw01, bins=10, range=(0, 1))
    zero_share = float((raw01 == 0).mean())
    first = course.order(0)[:SHOWN]
    again = course.order(1)[:SHOWN]
    sizes = [len(course.batch_indices(s)) for s in range(course.EPOCH_STEPS)]
    if sum(sizes) != course.TRAIN or sizes[-1] != course.LAST_BATCH:
        raise ValueError("the batches do not cover the training set")
    labels = [[int(v) for v in y[course.batch_indices(s)][:SHOWN]] for s in range(3)]
    belt = list(range(BELT)) + [course.EPOCH_STEPS - 1]      # the batches on the belt
    meta = dict(
        source="MNIST as examples/mnist_blocks.py loads it; the shuffled order of "
               "voxanim.study_course (seeded)", evidence="exact arithmetic on the data",
        train=course.TRAIN, test=course.TEST, label=int(y[DIGIT]), index=DIGIT,
        zoom=[int(r0), int(c0)], raw=img255[sl].tolist(), tensor=img01[sl].tolist(),
        normalised=norm[sl].tolist(), pixel=[int(pi), int(pj)], pixel_raw=int(v),
        pixel_tensor=float(p01), pixel_norm=float(pn),
        pixel_sum="(%.3f − %.4f) ÷ %.4f" % (p01, course.MEAN, course.STD),
        mean=stats["mean"], std=stats["std"], script_mean=course.MEAN,
        script_std=course.STD, pixels=int(raw01.size), paper=stats["low"], ink=stats["high"],
        zero_share=zero_share, hist=[int(h) for h in hist], edges=[float(e) for e in edges],
        batch=course.BATCH, steps=course.EPOCH_STEPS, last_batch=course.LAST_BATCH,
        full_batches=course.EPOCH_STEPS - 1, first=[int(i) for i in first],
        again=[int(i) for i in again], batch_labels=labels, belt=belt,
        belt_sizes=[sizes[b] for b in belt],
        belt_digits=[[int(i) for i in course.batch_indices(b)[:4]] for b in belt],
        epoch_sum="%d × %d + %d = %s" % (course.EPOCH_STEPS - 1, course.BATCH,
                                         course.LAST_BATCH, format(course.TRAIN, ",")),
        first_shape=shape(course.first_batch()["x"].shape),
        first_printed=torch_size(course.first_batch()["x"].shape),
        data=list(code_lines("data")), loader=list(code_lines("loader")),
        limitations=["the shuffled order is the numpy twin's (voxanim.study_course.order), "
                     "not PyTorch's: a run of the script draws other indices",
                     "the mean and spread are of all 47,040,000 training pixels, in float64; "
                     "the script uses them rounded to 4 places"])
    panels = (
        Panel("train_set[0]", "the first training digit, pixels 0..255", "images",
              dict(values=img01[None].astype(float), labels=["label %d" % y[DIGIT]])),
        Panel("Three stages of one window", "a %d x %d window: 0..255, then ToTensor (÷ 255), "
              "then Normalize" % (PATCH, PATCH), "matrix",
              dict(values=np.hstack([img255[sl], img01[sl], norm[sl]]).astype(float))),
        Panel("Every training pixel", "how many of the %s pixels have each brightness "
              "(mean %.5f, spread %.5f)" % (format(raw01.size, ","), stats["mean"], stats["std"]),
              "bars", dict(values=hist.astype(float),
                           labels=["%.1f" % e for e in edges[:-1]])),
        Panel("The batches", "the size of each of the 938 batches of one epoch", "bars",
              dict(values=np.array(sizes, float), labels=[str(i) for i in range(len(sizes))])),
    )
    return Study("The data pipeline", "Dataset, ToTensor and Normalize, then a shuffling "
                 "DataLoader: 60,000 digits become 938 batches", panels, meta)


STUDIES = {"blocks-logits-and-softmax": logits_and_softmax,
           "blocks-which-is-which": which_is_which,
           "blocks-the-data-pipeline": data_pipeline}

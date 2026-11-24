"""The counts behind animation 183, ``counting-weights-dnn-and-cnn``: how many numbers a
dense network and a convolutional network learn, layer by layer.

The two models are the ones met elsewhere on the page:

* the DNN of "An MLP, step by step" (:func:`voxanim.mnist.train`, sizes
  :data:`voxanim.mnist.SIZES` = 784-64-32-10), written in PyTorch as ``nn.Flatten()``,
  ``nn.Linear(784, 64)``, ``nn.ReLU()``, ``nn.Linear(64, 32)``, ``nn.ReLU()``,
  ``nn.Linear(32, 10)``;
* the CNN of "A CNN training step" (:func:`voxanim.mnist.cnn_train`: a 3 x 3 convolution
  with 4 filters, ReLU, 2 x 2 max-pool, a dense layer to 10), written as
  ``nn.Conv2d(1, 4, kernel_size=3)``, ``nn.ReLU()``, ``nn.MaxPool2d(2)``, ``nn.Flatten()``,
  ``nn.Linear(4 * 13 * 13, 10)``.

A dense layer has ``inputs x outputs`` weights and ``outputs`` biases; a convolution
``out_channels x in_channels x k x k`` weights and ``out_channels`` biases, whatever the
picture's size; ReLU, pooling and flattening have none. Every count is exact arithmetic on
the layers' shapes, and each is checked against the ``.size`` of the parameter arrays of
the repository's own models (trained, as the page shows them, and freshly initialised);
the filters' independence of the picture's size is checked by running the trained filters
over a bigger picture. Nothing here imports torch: the code lines are text.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import mnist
from .study import Panel, Study

SIDE = 28                               # an MNIST picture is SIDE x SIDE, one grey channel
CHANNELS = 1
SIZES = tuple(mnist.SIZES)              # the DNN: 784-64-32-10
FILTERS = mnist.CNN_FILTERS             # the CNN: 4 filters ...
KERNEL = SIDE - mnist.CONV_HW + 1       # ... of 3 x 3 (a valid convolution: 28 -> 26)
POOL = mnist.CONV_HW // mnist.POOL_HW   # 2 x 2 max-pool: 26 -> 13
CLASSES = mnist.CLASSES
TOY = (3, 2)                            # the small layer that shows the rule
BIGGER = 84                             # a bigger picture the same filters read
CODE = "sum(p.numel() for p in model.parameters())"
DNN_FROM = "An MLP, step by step"
CNN_FROM = "A CNN training step"
EVIDENCE = "exact arithmetic: weights counted from the layers' shapes, checked against " \
           "the parameter arrays of the page's own two models"


# ====================================================================== one layer
def linear(n_in, n_out, code=None):
    """A dense layer: one weight for every input of every output, one bias per output."""
    w, b = n_in * n_out, n_out
    return dict(kind="Linear", code=code or "nn.Linear(%d, %d)" % (n_in, n_out),
                shape=[n_in, n_out], inputs=n_in, outputs=n_out, weights=w, biases=b,
                total=w + b)


def conv(in_channels, out_channels, kernel):
    """A convolution: each of ``out_channels`` filters has ``in_channels x kernel x kernel``
    weights and one bias. The picture's size does not enter."""
    w, b = out_channels * in_channels * kernel * kernel, out_channels
    return dict(kind="Conv2d", code="nn.Conv2d(%d, %d, kernel_size=%d)"
                % (in_channels, out_channels, kernel),
                shape=[out_channels, in_channels, kernel, kernel], in_channels=in_channels,
                out_channels=out_channels, kernel=kernel, weights=w, biases=b, total=w + b)


def nothing(kind, code):
    """A layer with nothing to learn: it reshapes, clips or picks."""
    return dict(kind=kind, code=code, shape=[], weights=0, biases=0, total=0)


def with_running(layers):
    """``layers`` with ``running``: the total so far, after each one."""
    run = 0
    for layer in layers:
        run += layer["total"]
        layer["running"] = run
    return layers


def dnn_layers(sizes=SIZES):
    """The DNN's code lines, in order, with their counts."""
    layers = [nothing("Flatten", "nn.Flatten()")]
    for k, (a, b) in enumerate(itertools.pairwise(sizes)):
        if k:
            layers.append(nothing("ReLU", "nn.ReLU()"))
        layers.append(linear(a, b))
    return with_running(layers)


def pooled_side(side=SIDE, kernel=KERNEL, pool=POOL):
    """The side of each map after the convolution (no padding) and the pooling."""
    return (side - kernel + 1) // pool


def cnn_layers(side=SIDE, filters=FILTERS, kernel=KERNEL, pool=POOL, classes=CLASSES):
    """The CNN's code lines, in order, with their counts."""
    p = pooled_side(side, kernel, pool)
    dense = linear(filters * p * p, classes, "nn.Linear(%d * %d * %d, %d)"
                   % (filters, p, p, classes))
    dense.update(maps=filters, map_side=p)
    return with_running([conv(CHANNELS, filters, kernel), nothing("ReLU", "nn.ReLU()"),
                         nothing("MaxPool2d", "nn.MaxPool2d(%d)" % pool),
                         nothing("Flatten", "nn.Flatten()"), dense])


def learned(layers):
    """The layers that learn something, in order."""
    return [layer for layer in layers if layer["total"]]


# ====================================================================== the real arrays
def check(layers, arrays, model):
    """Each learning layer's weights and biases against the ``.size`` of its parameter
    arrays ``[[W, b], ...]``; the arrays' sizes, as counted."""
    sizes = [[int(np.asarray(w).size), int(np.asarray(b).size)] for w, b in arrays]
    want = [[layer["weights"], layer["biases"]] for layer in learned(layers)]
    if sizes != want:
        raise ValueError("%s: counted %s, the arrays hold %s" % (model, want, sizes))
    return sizes


def numel(arrays):
    """Every number in ``[[W, b], ...]``: ``sum(p.numel() for p in model.parameters())``."""
    return int(sum(np.asarray(a).size for pair in arrays for a in pair))


def dnn_arrays():
    """The DNN's parameters as the page shows them, trained (cached), and freshly
    initialised by :func:`voxanim.mnist._init`: ``(trained, initialised)``."""
    return mnist.train()[0], mnist._init(np.random.default_rng(0))


def cnn_arrays():
    """The CNN's trained parameters, as "A CNN training step" shows them:
    ``[[Wc (3, 3, F), bc], [Wd (676, 10), bd]]``."""
    return mnist.cnn_train()[0]


def filter_maps(Wc, bc, picture):
    """The trained filters run over ``picture`` (``side x side``), no padding:
    ``(side - k + 1, side - k + 1, F)`` maps. Any side works with the same numbers."""
    Wc = np.asarray(Wc, float)
    k = Wc.shape[0]
    win = np.lib.stride_tricks.sliding_window_view(np.asarray(picture, float), (k, k))
    return np.einsum("ijab,abf->ijf", win, Wc) + np.asarray(bc, float)


def any_size(Wc, bc, sides=(SIDE, BIGGER)):
    """For each picture side: the maps the filters give and the numbers they used."""
    rng = np.random.default_rng(183)
    out = []
    for s in sides:
        maps = filter_maps(Wc, bc, rng.random((s, s)))
        out.append(dict(side=int(s), map_side=int(maps.shape[0]), maps=int(maps.shape[2]),
                        parameters=int(np.asarray(Wc).size + np.asarray(bc).size),
                        dense_inputs=int(maps.shape[2] * (maps.shape[0] // POOL) ** 2)))
    return out


def biggest(layers):
    """The index (into ``layers``) of the layer holding the most numbers."""
    return int(np.argmax([layer["total"] for layer in layers]))


# ====================================================================== the study
def counting_weights():
    dnn, cnn = dnn_layers(), cnn_layers()
    trained, fresh = dnn_arrays()
    dnn_sizes = check(dnn, trained, "the DNN, trained")
    check(dnn, fresh, "the DNN, initialised")
    cp = cnn_arrays()
    cnn_sizes = check(cnn, cp, "the CNN, trained")
    (Wc, bc), (Wd, _bd) = cp
    if Wd.shape != (cnn[-1]["inputs"], CLASSES) or Wc.shape != (KERNEL, KERNEL, FILTERS):
        raise ValueError("the CNN's arrays are not 3 x 3 x %d and %d x %d"
                         % (FILTERS, cnn[-1]["inputs"], CLASSES))
    toy = linear(*TOY)
    check([toy], mnist._init(np.random.default_rng(0), TOY), "the small layer")
    one = linear(1, 1)
    sizes = any_size(Wc, bc)
    d_total, c_total = dnn[-1]["running"], cnn[-1]["running"]
    filters = np.asarray(Wc, float).transpose(2, 0, 1)            # (F, 3, 3)
    for name, arrays, total in (("DNN", trained, d_total), ("CNN", cp, c_total)):
        if numel(arrays) != total:
            raise ValueError(name + ": the arrays do not add up to the running total")
    panels = (
        Panel("The DNN, layer by layer", "numbers each line of the 784-64-32-10 MLP learns: "
              "inputs x outputs weights plus one bias per output; Flatten and ReLU learn none",
              "bars", dict(values=np.array([x["total"] for x in dnn], float),
                           running=np.array([x["running"] for x in dnn], float),
                           labels=[x["code"] for x in dnn])),
        Panel("The CNN, layer by layer", "numbers each line of the CNN learns: 4 filters of "
              "1 x 3 x 3 plus 4 biases, then a dense layer from 4 x 13 x 13 to 10; ReLU, "
              "pooling and Flatten learn none", "bars",
              dict(values=np.array([x["total"] for x in cnn], float),
                   running=np.array([x["running"] for x in cnn], float),
                   labels=[x["code"] for x in cnn])),
        Panel("The CNN's four 3 x 3 filters", "the 36 trained weights of the CNN's filters, "
              "9 to a filter, as \"A CNN training step\" shows them", "images",
              dict(values=filters, biases=np.asarray(bc, float),
                   labels=["filter %d" % (k + 1) for k in range(FILTERS)])),
        Panel("Two totals", "numbers each model learns, counted from its layers' shapes and "
              "checked against its parameter arrays", "bars",
              dict(values=np.array([d_total, c_total], float), labels=["DNN", "CNN"])),
    )
    return Study(
        "Counting the weights: a DNN and a CNN", "Each dense output has one weight per input "
        "plus one bias; a filter has its 3 x 3 weights plus one bias, whatever the picture",
        panels,
        dict(source="the shapes of the page's two MNIST models, checked against their "
                    "parameter arrays (mnist.train and mnist.cnn_train, trained, as the page "
                    "shows them; the MLP also freshly initialised)",
             evidence=EVIDENCE, dataset="none: exact arithmetic on the layers' shapes",
             dnn_from=DNN_FROM, cnn_from=CNN_FROM, picture=[SIDE, SIDE], channels=CHANNELS,
             one=one, toy=toy,
             dnn=dict(sizes=list(SIZES), layers=dnn, total=d_total, array_sizes=dnn_sizes,
                      running=[x["running"] for x in learned(dnn)], biggest=biggest(dnn)),
             cnn=dict(filters=FILTERS, kernel=KERNEL, pool=POOL, layers=cnn, total=c_total,
                      array_sizes=cnn_sizes, running=[x["running"] for x in learned(cnn)],
                      biggest=biggest(cnn), any_size=sizes),
             code=CODE, ratio=d_total / c_total,
             shares=dict(dnn=dnn[biggest(dnn)]["total"] / d_total,
                         cnn=cnn[biggest(cnn)]["total"] / c_total),
             limitations=["the PyTorch lines are the same layers written in PyTorch; the "
                          "models themselves are numpy, and nothing here imports torch",
                          "the numpy CNN flattens its pooled maps channel last, PyTorch "
                          "channel first: the order differs, the count does not"]))


STUDIES = {"counting-weights-dnn-and-cnn": counting_weights}

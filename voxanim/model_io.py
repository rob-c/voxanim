"""Load (and save) fully-connected models so any pre-trained MNIST network can
be animated.

The bundled ``models/mnist_dnn_100x3.npz`` is reproducible: ``python3 -m
voxanim.model_io --make-default`` retrains it (:func:`make_default`).

A model is a list of ``(W, b)`` layers with ``W`` shaped ``(inputs, outputs)``;
ReLU between layers, softmax on the last.  Accepted files:

``.npz``   arrays ``W0, b0, W1, b1, ...`` (what :func:`save_dense` writes), or any
           names -- weights and biases are paired in order, see below.
           Optional ``preprocess`` (string array) records the input scaling.
``.pt``    a PyTorch ``state_dict`` (or a whole ``nn.Module``) -- needs torch.
``.pth``
``.h5``    Keras weights -- needs h5py.
``.keras``
``.json``  ``{"layers": [{"W": [[...]], "b": [...]}, ...], "preprocess": "unit"}``

Pairing: arrays are taken in file order (names sorted naturally, so
``layers.10`` comes after ``layers.9``); every 2-D array is a weight matrix
and the 1-D array after it, if its length fits, is its bias.  Orientation is
inferred from the chain of sizes starting at 784 inputs, so PyTorch's
``(out, in)`` and Keras' ``(in, out)`` both work.

Input preprocessing (how the network expects pixels):

``unit``      pixels in 0..1 (default)
``standard``  ``(x - 0.1307) / 0.3081`` -- the usual PyTorch MNIST transform
``centered``  ``2x - 1``
``raw``       0..255
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings

import numpy as np

PREPROCESS = {
    "unit": lambda x: x,
    "standard": lambda x: (x - 0.1307) / 0.3081,
    "centered": lambda x: 2.0 * x - 1.0,
    "raw": lambda x: x * 255.0,
}


DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                             "mnist_dnn_100x3.npz")
# How DEFAULT_MODEL was made: :func:`voxanim.mnist.train` with these settings.
DEFAULT_RECIPE = dict(sizes=(784, 100, 100, 100, 10), epochs=6, batch=128, lr=1e-3, seed=0)


def _natural(key):
    """Sort key that orders digit runs numerically (``fc.9`` before ``fc.10``)."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", key)]


def _ordered(named):
    """``[(name, array)]`` -> arrays layer by layer, weight before bias.

    Names sort naturally by their layer prefix (``fc.10`` after ``fc.9``); within
    a layer the 2-D weight comes first even when ``bias`` sorts before
    ``weight`` / ``kernel`` alphabetically (PyTorch, Keras)."""
    def key(item):
        name, a = item
        parent = name.rsplit(".", 1)[0] if "." in name else name.rsplit("/", 1)[0]
        return (_natural(parent), -np.ndim(a))
    return [a for _n, a in sorted(named, key=key)]


def _pair(arrays, n_in=784):
    """Ordered arrays -> ``[(W (in, out), b)]`` following the chain of sizes.

    The storage convention is a property of the file, not of each layer: a
    square hidden layer (100 x 100) cannot tell (in, out) from (out, in) by
    its shape.  So it is decided once, from the first weight that is not
    square -- PyTorch ``nn.Linear`` stores (out, in), Keras/numpy (in, out) --
    and then applied to every layer."""
    arrays = [np.asarray(a, np.float64) for a in arrays]
    mats = [a for a in arrays if a.ndim == 2]
    if not mats:
        raise ValueError("no dense layers found")
    transpose = None
    width = n_in
    for a in mats:                       # walk the chain until a shape decides it
        if a.shape[0] != a.shape[1]:
            if a.shape[0] == width:
                transpose = False
            elif a.shape[1] == width:
                transpose = True
            break
        width = a.shape[1]
    if transpose is None:
        transpose = False                # all square: assume (in, out)
    layers = []
    width = n_in
    i = 0
    while i < len(arrays):
        a = arrays[i]
        if a.ndim != 2:
            i += 1                       # stray scalars / norms are not dense layers
            continue
        W = a.T if transpose else a
        if W.shape[0] != width:
            raise ValueError("layer %d: weight %s (%s) does not take %d inputs"
                             % (len(layers) + 1, a.shape, "out, in" if transpose else
                                "in, out", width))
        b = np.zeros(W.shape[1])
        if i + 1 < len(arrays) and arrays[i + 1].ndim == 1 and len(arrays[i + 1]) == W.shape[1]:
            b = arrays[i + 1]
            i += 1
        layers.append((W, b))
        width = W.shape[1]
        i += 1
    return layers


def _torch_load(path):
    """A PyTorch file's contents, safely if possible.

    ``weights_only=True`` reads tensors without running pickled code; a whole
    pickled ``nn.Module`` needs the unsafe path, so that is only tried after,
    with a warning (only load such files from sources you trust).
    """
    import torch  # optional dependency
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:            # torch raises UnpicklingError or RuntimeError
        warnings.warn("%s is not a plain state_dict (%s); loading it with "
                      "weights_only=False, which runs code stored in the file"
                      % (path, e), stacklevel=3)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_dense(path, n_in=784):
    """``(layers, preprocess)`` from a model file (see the module docstring).

    ``layers`` is ``[(W (in, out), b (out,)), ...]`` in float64 and
    ``preprocess`` a key of :data:`PREPROCESS`.
    """
    ext = os.path.splitext(path)[1].lower()
    pre = "unit"
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as d:
            keys = [k for k in d.files if k != "preprocess"]
            if all(re.fullmatch(r"[Wb]\d+", k) for k in keys):
                n = len(keys) // 2
                arrays = [d[k] for i in range(n) for k in ("W%d" % i, "b%d" % i)]
            else:
                arrays = _ordered([(k, d[k]) for k in keys])
            if "preprocess" in d.files:
                pre = str(d["preprocess"])
    elif ext in (".pt", ".pth"):
        obj = _torch_load(path)
        if hasattr(obj, "state_dict"):
            obj = obj.state_dict()
        arrays = _ordered([(k, v.detach().cpu().numpy()) for k, v in obj.items()])
    elif ext in (".h5", ".keras"):
        import h5py  # optional dependency
        named = []
        with h5py.File(path, "r") as f:
            f.visititems(lambda name, obj: named.append((name, np.array(obj)))
                         if hasattr(obj, "shape") and obj.ndim in (1, 2) else None)
        arrays = _ordered(named)
    elif ext == ".json":
        with open(path) as f:
            d = json.load(f)
        arrays = [a for layer in d["layers"] for a in (layer["W"], layer.get("b",
                                                                           []))]
        arrays = [np.asarray(a) for a in arrays if len(a)]
        pre = d.get("preprocess", "unit")
    else:
        raise ValueError("unsupported model file: %s" % path)
    if pre not in PREPROCESS:
        raise ValueError("unknown preprocess %r (one of %s)" % (pre, ", ".join(PREPROCESS)))
    return _pair(arrays, n_in), pre


def save_dense(path, layers, preprocess="unit"):
    """Write ``layers`` as float32 ``W0, b0, W1, b1, ...`` plus ``preprocess`` to ``path``."""
    arrays = {}
    for n, (W, b) in enumerate(layers):
        arrays["W%d" % n] = np.asarray(W, np.float32)
        arrays["b%d" % n] = np.asarray(b, np.float32)
    np.savez(path, preprocess=np.array(preprocess), **arrays)


def forward(layers, x):
    """Run a dense ReLU network on ``x`` ``(784,)`` or a batch ``(N, 784)``.

    Returns ``(zs, acts, p)``: the pre-activation of every layer, the input
    of every layer (``acts[0]`` is ``x``; ReLU between layers) and the softmax
    of the last pre-activation, normalised along the last axis.
    """
    zs, acts = [], [np.asarray(x, np.float64)]
    for n, (W, b) in enumerate(layers):
        z = acts[-1] @ W + b
        zs.append(z)
        if n < len(layers) - 1:
            acts.append(np.maximum(z, 0.0))
    e = np.exp(zs[-1] - zs[-1].max(axis=-1, keepdims=True))
    return zs, acts, e / e.sum(axis=-1, keepdims=True)


def make_default(path=DEFAULT_MODEL):
    """Retrain the bundled 784-100-100-100-10 network and write it to ``path``.

    Deterministic: the arrays equal the committed ``models/mnist_dnn_100x3.npz``.
    Returns the test accuracy (0.9755).
    """
    from . import mnist
    params, acc = mnist.train(**DEFAULT_RECIPE)
    save_dense(path, params, "unit")
    return acc


def main(argv=None):
    """``python3 -m voxanim.model_io --make-default [--out PATH]``."""
    ap = argparse.ArgumentParser(prog="python3 -m voxanim.model_io", description=main.__doc__)
    ap.add_argument("--make-default", action="store_true",
                    help="retrain the bundled model (%s)" % os.path.basename(DEFAULT_MODEL))
    ap.add_argument("--out", default=DEFAULT_MODEL, help="where to write it")
    args = ap.parse_args(argv)
    if not args.make_default:
        ap.error("nothing to do (pass --make-default)")
    print("wrote %s (test accuracy %.4f)" % (args.out, make_default(args.out)))


if __name__ == "__main__":
    main()

"""Hardware acceleration: training and rendering on a GPU (Metal or CUDA)
through PyTorch, when it is installed.

voxanim's models are written in numpy (float64), and the numpy code stays
the reference: every model still has its numpy forward and hand-written
backward pass, and the animations draw from numpy arrays.  This module adds
a second way to *train* the expensive models, and to composite large frames:

* :func:`device` picks where work runs, from ``$VOXANIM_DEVICE``:
  ``auto`` (the default), ``cuda``, ``mps``, ``cpu`` (PyTorch on the CPU),
  or ``numpy`` (no PyTorch at all: the original code paths, bit for bit).
  ``auto`` is *the fastest measured path*, not "always the GPU": a caller
  says whether its work is worth moving (``prefer``), and ``auto`` sends it
  to the GPU (CUDA, else Metal) only then, keeping everything else on numpy.
  Measured on an M5 Max: the convolutional autoencoder trains 9x faster on
  Metal, an MLP step with a batch of 1,024 2.7x faster, but the batch-64 MLPs
  most animations train are 1.5-5x *slower* through PyTorch (per-step
  overhead beats the arithmetic) -- they stay numpy, bit for bit.  The
  explicit modes force their device for everything.  Without PyTorch
  installed every path is the numpy one.
* Rendering: on an Apple GPU every frame is composited by one Metal kernel
  (:mod:`voxanim.raster_metal`, chosen by :func:`voxanim.scene.framebuffer`);
  elsewhere the generic tensor framebuffer (:mod:`voxanim.raster_torch`)
  serves big frames.  ``python3 -m voxanim gpucheck`` compares either with
  numpy, frame by frame.
* :func:`fit` runs a model's minibatch loop on that device in float32 (Apple
  GPUs have no float64) with the same Adam as :class:`voxanim.optim.Adam`.
  The initial weights and the minibatch order still come from the model's
  seeded numpy generator, so an accelerated run is the *same experiment*
  as the numpy one -- the same initial weights, the same batches in the same
  order -- and differs only by float32 rounding: the trained weights agree
  to a tolerance, not bit for bit.

The cache (:func:`voxanim.cache.cached_npz`) keys trained weights on their
settings, not on the device: weights already in the cache (or unpacked from
a release's ``voxanim-weights.tar.gz``) are used as they are, whichever
device would have trained them, so earlier artefacts stay valid.  Each new
cache file records the device that trained it (:func:`describe`).
"""

from __future__ import annotations

import contextlib
import itertools
import os

import numpy as np

ENV = "VOXANIM_DEVICE"
FLOAT_ENV = "VOXANIM_FLOAT"           # "32" (default) or "64" (CPU / CUDA: for checking a port)
CHOICES = ("auto", "cuda", "mps", "cpu", "numpy")

_torch = None
_tried = False


def torch():
    """The ``torch`` module, or None when it is not installed (or disabled)."""
    global _torch, _tried
    if os.environ.get(ENV, "auto").lower() == "numpy":
        return None                       # decided per call: the mode can change (forced())
    if not _tried:
        _tried = True                     # import once, and only when a torch mode asks
        try:
            import torch as t
            _torch = t
        except ImportError:
            _torch = None
    return _torch


def choice():
    """``$VOXANIM_DEVICE``, checked."""
    c = os.environ.get(ENV, "auto").lower()
    if c not in CHOICES:
        raise ValueError("%s=%s: one of %s" % (ENV, c, ", ".join(CHOICES)))
    return c


def device(prefer=True):
    """The ``torch.device`` work runs on, or None for numpy.  ``prefer`` is the
    caller's verdict on whether its work gains from a GPU; under ``auto`` work
    that does not gain, or with no GPU to go to, stays on numpy."""
    t = torch()
    c = choice()
    if t is None or c == "numpy":
        if c in ("cuda", "mps", "cpu"):
            raise RuntimeError("%s=%s needs PyTorch (pip install voxanim[gpu])" % (ENV, c))
        return None
    if c == "auto":
        if not prefer:
            return None
        if t.cuda.is_available():
            return t.device("cuda")
        if t.backends.mps.is_available():
            return t.device("mps")
        return None                     # PyTorch on the CPU is no faster than numpy here
    if c == "cuda" and not t.cuda.is_available():
        raise RuntimeError("%s=cuda but no CUDA device is available" % ENV)
    if c == "mps" and not t.backends.mps.is_available():
        raise RuntimeError("%s=mps but Metal is not available" % ENV)
    return t.device(c)


def enabled(prefer=True):
    """True when work of this kind (``prefer``, as for :func:`device`) runs
    through PyTorch."""
    return device(prefer) is not None


def describe(prefer=True):
    """``"numpy"`` or ``"torch-<version>/<device>"``: where such work runs."""
    dev = device(prefer)
    return "numpy" if dev is None else "torch-%s/%s" % (torch().__version__, dev.type)


TRAIN_WORK = 2e8          # multiply-adds per training step from which a GPU pays (measured)
RENDER_WIDTH = 7680       # output width from which the GPU framebuffer pays (measured: 8K
#                           conv2d 2.8x; at 4K diagrams gain 1.7x but sheets and voxels lose)


def mlp_work(sizes, batch):
    """Multiply-adds of one forward pass of a dense network (``sizes``) on a batch."""
    return batch * sum(a * b for a, b in itertools.pairwise(sizes))


@contextlib.contextmanager
def forced(name):
    """Run a block with ``$VOXANIM_DEVICE`` set to ``name`` (tests, benchmarks)."""
    old = os.environ.get(ENV)
    os.environ[ENV] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


# ------------------------------------------------------------------ arrays


def float_dtype():
    """The float type accelerated training uses: float32, or float64 when
    ``$VOXANIM_FLOAT=64`` (not on Apple GPUs, which have no float64)."""
    t = torch()
    if os.environ.get(FLOAT_ENV, "32") == "64":
        if choice() == "mps":
            raise RuntimeError("%s=64: Apple GPUs (mps) have no float64" % FLOAT_ENV)
        return t.float64
    return t.float32


def to_device(a, dev=None, dtype=None):
    """A numpy array as a tensor of :func:`float_dtype` on ``dev`` (default
    :func:`device` of preferred work)."""
    t = torch()
    dev = device() if dev is None else dev
    return t.as_tensor(np.asarray(a), dtype=dtype or float_dtype(), device=dev)


def to_numpy(x, dtype=np.float64):
    """A tensor (or a dict of them) back as numpy ``dtype`` arrays."""
    if isinstance(x, dict):
        return {k: to_numpy(v, dtype) for k, v in x.items()}
    return x.detach().to("cpu").numpy().astype(dtype)


def sync():
    """Wait for queued GPU work (for timing)."""
    t, dev = torch(), device()
    if t is None:
        return
    if dev is None:
        return
    if dev.type == "cuda":
        t.cuda.synchronize()
    elif dev.type == "mps":
        t.mps.synchronize()


# ------------------------------------------------------------------ training


class Adam:
    """:class:`voxanim.optim.Adam`'s update on tensors (float32):

        m = 0.9 m + 0.1 g;  v = 0.999 v + 0.001 g^2
        p = p - lr * (m / (1 - 0.9^t)) / (sqrt(v / (1 - 0.999^t)) + 1e-8)
    """

    def __init__(self, params, lr):
        t = torch()
        self.lr, self.t = lr, 0
        self.m = {k: t.zeros_like(v) for k, v in params.items()}
        self.v = {k: t.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        """One update of every parameter in place."""
        from .optim import BETA1, BETA2, EPS, GAIN1, GAIN2
        t = torch()
        self.t += 1
        lr = self.lr if lr is None else lr
        c1, c2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        with t.no_grad():
            for k, p in params.items():
                g = grads[k]
                self.m[k].mul_(BETA1).add_(g, alpha=GAIN1)
                self.v[k].mul_(BETA2).addcmul_(g, g, value=GAIN2)
                p.sub_(lr * (self.m[k] / c1) / (t.sqrt(self.v[k] / c2) + EPS))


def params_to_device(P, dev=None, dtype=None):
    """numpy parameters as (float32) leaf tensors that require gradients."""
    return {k: to_device(v, dev, dtype).requires_grad_(True) for k, v in P.items()}


def fit(P, loss_fn, batches, lr, after_step=None, data=None, dtype=None, dev=None):
    """Train parameters ``P`` (numpy, updated in place at the end) with Adam on
    :func:`device`.

    ``batches`` yields index arrays (from :func:`voxanim.minibatch.minibatches`,
    drawn from the model's numpy generator, so the order is the numpy run's);
    ``data`` is the training array, uploaded once; ``loss_fn(Pt, xb, i)``
    returns the scalar loss of batch ``xb`` (rows of ``data``; ``i`` is the
    0-based step) from the tensor parameters ``Pt``.  ``after_step(step, Pt)``
    runs after each update with the 1-based step count (snapshots).  ``dtype``
    overrides float32 (the tests run the port in float64 on the CPU against
    the numpy model).  Returns the tensor parameters too.
    """
    t = torch()
    dev = device() if dev is None else dev
    Pt = params_to_device(P, dev, dtype)
    opt = Adam(Pt, lr)
    xd = to_device(data, dev, dtype)
    for i, idx in enumerate(batches):
        xb = xd[t.as_tensor(np.asarray(idx), device=dev)]
        loss = loss_fn(Pt, xb, i)
        grads = t.autograd.grad(loss, list(Pt.values()))
        opt.step(Pt, dict(zip(Pt.keys(), grads)))
        if after_step is not None:
            after_step(i + 1, Pt)
    P.update(to_numpy(Pt))
    return Pt

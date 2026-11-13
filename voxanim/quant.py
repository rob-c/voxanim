"""Quantization and reduced precision, in plain numpy, on real trained models.

Everything the three ``efficiency`` animations show is computed here, on the
MNIST MLP of :mod:`voxanim.mnist` (784-64-32-10, 96.8% test accuracy) and
the deep MLP of :mod:`voxanim.deepnet`:

* **Uniform quantizers** (:class:`Quantizer`): ``q = clip(round(x / s) + z)``
  on ``2^bits`` integer codes, symmetric (``z = 0``, codes ``-qmax .. qmax``
  with ``qmax = 2^(bits-1) - 1``: int8 has 255 levels, int4 15, 2 bits the
  ternary ``{-s, 0, s}``) or asymmetric (codes ``0 .. 2^bits - 1`` with a
  zero point), with one scale per tensor or one per channel
  (:func:`calibrate`).  **Calibration** of the scale: min-max (the largest
  value maps to the last code) or the clip fraction of the range that
  minimises the squared quantization error, searched over :data:`CLIP_GRID`
  (:func:`calibrate_mse`).  :func:`quantize_params` applies one scheme to
  every weight matrix of the MLP; :func:`ptq_accuracies` measures the test
  accuracy of each scheme at each bit width (post-training quantization);
  :func:`activation_ranges` / :func:`forward_quantized` also quantize the
  activations at inference; :func:`model_bytes` is the size on disk.
* **Quantization-aware training** (:func:`qat_train`): the forward pass uses
  the fake-quantized weights ``Q(W)`` (rounded, then dequantized, so the
  network computes with the values an integer kernel would), the backward
  pass uses the **straight-through estimator**: ``dL/dW := dL/dQ(W)`` inside
  the clipping range and 0 outside (:meth:`Quantizer.ste_mask`), as if the
  rounding had derivative 1.  The true derivative of ``round`` is 0 almost
  everywhere, so this gradient cannot be checked against finite differences
  of the quantized loss; ``tests/test_quant.py`` checks it against the
  definition instead, and the bias gradients (which rounding does not touch)
  against finite differences.  The scales are calibrated once, from the
  trained float weights, and kept fixed.  Runs start from the trained MLP
  and record the latent weights of one layer, the quantized network's test
  accuracy and every minibatch loss.
* **Reduced precision** (:func:`round_to`, :func:`precision_train`): float16
  through numpy's own ``float16``; **bfloat16 emulated** by rounding a
  float32 to its nearest value with a 7-bit mantissa (round half to even on
  the upper 16 bits, :func:`round_bf16`; numpy has no bfloat16).  A training
  run in a format rounds every tensor an operation produces -- the weights
  as used, activations, gradients, and the optimiser's moments unless the
  **master weights** pattern keeps those in float32 -- while sums and
  products accumulate in float32, as GPU tensor cores do.  Loss scaling
  multiplies the loss (so every gradient) by a power of two before the cast
  and divides it out in float32.  :func:`underflow_gradient` computes a real
  weight gradient of a 20-layer tanh MLP (:data:`DEEP_SPEC`, weights of std
  0.05) whose values straddle float16's smallest subnormal, ``6×10⁻⁸``.

Every run is seeded and cached by :func:`voxanim.cache.cached_npz` under all
its settings; the numbers quoted in the animations are what these functions
return on the cached models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .deepnet import Spec, backward, forward, init, subset
from .minibatch import minibatches
from .mnist import CLASSES, _backward, _forward, _init, load, train
from .optim import BETA1, BETA2, EPS, GAIN1, GAIN2, Adam

BITS = (8, 4, 2)                                 # the bit widths the animations compare
CLIP_GRID = np.round(np.linspace(0.1, 1.0, 46), 2)   # clip fractions calibration searches
WATCH = 1                                        # the layer whose weights are drawn: W1, 64 x 32
QAT_EPOCHS, QAT_LR, QAT_BATCH = 2, 1e-3, 128     # the quantization-aware fine-tuning
PRECISION_EPOCHS, PRECISION_LR, PRECISION_BATCH = 2, 1e-3, 128
EVAL_EVERY = 117                                 # steps between test evaluations (8 per epoch)
CALIB_IMAGES = 1000                              # training images activation ranges are read from
LOSS_SCALE = 65536.0                             # 2^16: the usual initial loss scale
FORMATS = ("float32", "float16", "bfloat16")
FIELDS = {"float32": (1, 8, 23), "float16": (1, 5, 10), "bfloat16": (1, 8, 7)}   # bits per field
DEEP_SIZES = (784,) + (100,) * 20 + (10,)       # the initialisation animation's network
DEEP_STD = 0.05                                  # between its std 0.01 (hopeless) and Xavier's 0.1
DEEP_SPEC = Spec(DEEP_SIZES, "tanh", "normal", DEEP_STD)
DEEP_LAYER = 10                                  # the hidden layer whose weight gradient is shown
DEEP_BATCH = 64                                  # images the gradient is measured on


# ------------------------------------------------------------------ quantizers


def qrange(bits, symmetric=True):
    """``(qmin, qmax)``: the integer codes of a ``bits``-bit quantizer.
    Symmetric codes run ``-qmax .. qmax`` (one code unused), asymmetric
    ``0 .. 2^bits - 1``."""
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** bits - 1


@dataclass(frozen=True)
class Quantizer:
    """A uniform quantizer: ``q = clip(round(x / scale) + zero, qmin, qmax)``
    and back ``x' = (q - zero) * scale``.

    ``scale`` and ``zero`` broadcast against the tensor: shape ``(1, 1)`` for
    a per-tensor quantizer of a matrix, ``(1, out)`` for one scale per output
    channel.
    """

    bits: int
    scale: np.ndarray
    zero: np.ndarray
    qmin: int
    qmax: int

    @property
    def levels(self):
        """How many distinct values the quantizer can produce."""
        return self.qmax - self.qmin + 1

    @property
    def per_channel(self):
        """True when there is more than one scale."""
        return np.asarray(self.scale).size > 1

    def quantize(self, x):
        """Integer codes of ``x`` (as an int64 array)."""
        q = np.round(np.asarray(x, np.float64) / self.scale) + self.zero
        return np.clip(q, self.qmin, self.qmax).astype(np.int64)

    def dequantize(self, q):
        """The float values the codes ``q`` stand for."""
        return (np.asarray(q, np.float64) - self.zero) * self.scale

    def fake(self, x):
        """``x`` rounded to the quantizer's grid: what an integer kernel computes with."""
        return self.dequantize(self.quantize(x))

    def ste_mask(self, x):
        """The straight-through estimator's "derivative" of :meth:`fake` at
        ``x``: 1 inside the clipping range (the rounding is treated as the
        identity), 0 where the value saturates."""
        q = np.round(np.asarray(x, np.float64) / self.scale) + self.zero
        return ((q >= self.qmin) & (q <= self.qmax)).astype(np.float64)

    def grid(self):
        """The representable values of a per-tensor quantizer, ascending."""
        if self.per_channel:
            raise ValueError("grid() needs a per-tensor quantizer")
        return self.dequantize(np.arange(self.qmin, self.qmax + 1)).ravel()


def _reduce_axes(x, axis):
    """The axes a per-channel statistic is taken over (all but ``axis``)."""
    if axis is None:
        return None
    return tuple(i for i in range(x.ndim) if i != axis % x.ndim)


def calibrate(x, bits, symmetric=True, axis=None, clip=1.0):
    """Min-max calibration: the quantizer whose last code is ``clip`` times
    the largest magnitude of ``x`` (symmetric) or whose codes span ``clip``
    times ``[min(x, 0), max(x, 0)]`` (asymmetric: 0 is always representable
    exactly).  ``axis`` None gives one scale for the tensor; an integer gives
    one per index along that axis (per channel).  ``clip`` may be an array
    broadcasting against those scales."""
    x = np.asarray(x, np.float64)
    reduce = _reduce_axes(x, axis)
    qmin, qmax = qrange(bits, symmetric)
    if symmetric:
        amax = np.abs(x).max(axis=reduce, keepdims=True) * clip
        scale = np.where(amax > 0, amax / qmax, 1.0)
        zero = np.zeros_like(scale)
    else:
        lo = np.minimum(x.min(axis=reduce, keepdims=True), 0.0) * clip
        hi = np.maximum(x.max(axis=reduce, keepdims=True), 0.0) * clip
        scale = np.where(hi > lo, (hi - lo) / (qmax - qmin), 1.0)
        zero = np.round(-lo / scale)
    return Quantizer(bits, scale, zero, qmin, qmax)


def calibrate_mse(x, bits, symmetric=True, axis=None, grid=CLIP_GRID):
    """The clip fraction in ``grid`` (per tensor, or per channel along
    ``axis``) that minimises the summed squared error ``|Q(x) - x|^2``, and
    the quantizer built with it: ``(quantizer, clip)``.

    Clipping the range trades saturating the few largest values for a
    finer grid under the many small ones; at 2 bits that is the difference
    between chance and a working network."""
    x = np.asarray(x, np.float64)
    reduce = _reduce_axes(x, axis)
    errors = np.stack([((calibrate(x, bits, symmetric, axis, c).fake(x) - x) ** 2)
                       .sum(axis=reduce, keepdims=True) for c in grid])
    clip = np.asarray(grid)[errors.argmin(0)]
    return calibrate(x, bits, symmetric, axis, clip), clip


# ------------------------------------------------------------------ the MLP, quantized

SCHEMES = {                     # name -> how every weight matrix's scale is set
    "per-tensor": dict(per_channel=False, calibrated=False),
    "per-channel": dict(per_channel=True, calibrated=False),
    "calibrated": dict(per_channel=True, calibrated=True),
}
CHANNEL_AXIS = 1                # a dense weight ``W (in, out)``: a channel is an output column


def weight_quantizer(W, bits, per_channel=False, calibrated=False, symmetric=True):
    """The quantizer of one weight matrix under a scheme (see :data:`SCHEMES`)."""
    axis = CHANNEL_AXIS if per_channel else None
    if calibrated:
        return calibrate_mse(W, bits, symmetric, axis)[0]
    return calibrate(W, bits, symmetric, axis)


def quantize_params(params, bits, per_channel=False, calibrated=False, symmetric=True):
    """``(params_q, quantizers)``: every weight matrix of ``[[W, b], ...]``
    replaced by its fake-quantized value (biases stay in float)."""
    qs = [weight_quantizer(W, bits, per_channel, calibrated, symmetric) for W, _ in params]
    return [[q.fake(W), b] for (W, b), q in zip(params, qs, strict=True)], qs


def accuracy(params, x, y):
    """Fraction of ``(x, y)`` the MLP ``params`` classifies correctly."""
    return float((_forward(params, x)[2].argmax(1) == y).mean())


def ptq_accuracies(params, x, y, bits=BITS, schemes=SCHEMES):
    """Post-training quantization: ``{(bits, scheme): test accuracy}`` of the
    MLP with every weight matrix quantized under each scheme."""
    out = {}
    for b in bits:
        for name, kw in schemes.items():
            out[b, name] = accuracy(quantize_params(params, b, **kw)[0], x, y)
    return out


def activation_ranges(params, x_calib):
    """The largest input of each dense layer (the image, then each ReLU
    output) over the calibration images: an unsigned activation quantizer's
    range."""
    acts, _pre, _p = _forward(params, x_calib)
    return [float(a.max()) for a in acts]


def forward_quantized(params, x, bits, ranges):
    """Logits of the MLP with every dense layer's *input* quantized to
    ``bits`` unsigned codes over ``[0, range]`` (inputs are 0..1 pixels or
    ReLU outputs, so a zero point is not needed); ``params`` are typically
    already weight-quantized."""
    n = 2 ** bits - 1
    h = np.asarray(x, np.float64)
    z = None
    for k, (W, b) in enumerate(params):
        s = ranges[k] / n
        h = np.clip(np.round(h / s), 0, n) * s
        z = h @ W + b
        if k < len(params) - 1:
            h = np.maximum(z, 0.0)
    return z


def model_bytes(params, bits, per_channel=False):
    """Bytes to store the MLP with its weights packed at ``bits`` bits each;
    biases, and the quantizers' scales when ``bits < 32``, in float32."""
    n_w = sum(W.size for W, _ in params)
    n_b = sum(b.size for _, b in params)
    if bits >= 32:
        return 4 * (n_w + n_b)
    n_s = sum(W.shape[CHANNEL_AXIS] if per_channel else 1 for W, _ in params)
    return n_w * bits // 8 + 4 * (n_b + n_s)


def trained_mlp(cache_dir=None):
    """``(params, accuracy, (x_train, y_train, x_test, y_test))``: the MLP
    the ``mlp-*`` animations show, and MNIST."""
    params, acc = train(cache_dir=cache_dir)
    return params, acc, load(cache_dir)


# ------------------------------------------------------------------ quantization-aware training


def qat_forward_backward(P, quantizers, x, y_onehot):
    """One fake-quantized forward pass and its straight-through backward pass.

    ``P`` is the dict ``{"W0", "b0", ...}`` of latent float parameters.
    Returns ``(grads, grads_q, p)``: ``grads`` has ``P``'s keys, with
    ``dL/dW = dL/dQ(W) * ste_mask(W)``; ``grads_q[k]`` is ``dL/dQ(W_k)``, the
    gradient the float network has at the quantized weights; ``p`` the
    softmax output.
    """
    n = len(quantizers)
    params = [[quantizers[k].fake(P["W%d" % k]), P["b%d" % k]] for k in range(n)]
    acts, pre, p = _forward(params, x)
    grads, _deltas, _dacts = _backward(params, acts, pre, p, y_onehot)
    g = {}
    for k in range(n):
        dQ, db = grads[k]
        g["W%d" % k] = dQ * quantizers[k].ste_mask(P["W%d" % k])
        g["b%d" % k] = db
    return g, [grads[k][0] for k in range(n)], p


def qat_loss(P, quantizers, x, y_onehot):
    """Mean cross-entropy of the fake-quantized network (what the STE pretends
    to differentiate; exact for the biases)."""
    n = len(quantizers)
    params = [[quantizers[k].fake(P["W%d" % k]), P["b%d" % k]] for k in range(n)]
    p = _forward(params, x)[2]
    return float(-np.mean(np.log(np.sum(p * y_onehot, 1) + 1e-300)))


@dataclass
class QATRun:
    """A recorded quantization-aware fine-tuning run (see :func:`qat_train`)."""

    bits: int
    scales: np.ndarray           # (layers,) the fixed per-tensor scale of each weight matrix
    steps: np.ndarray            # (S,) the recorded steps (updates made so far; 0 = the PTQ start)
    W: np.ndarray                # (S, in, out) latent float weights of layer WATCH at those steps
    acc: np.ndarray              # (S,) test accuracy of the quantized network at those steps
    batch_loss: np.ndarray       # (steps,) cross-entropy of each minibatch before its update
    grad_q: np.ndarray           # (in, out) dL/dQ(W_WATCH) on the first minibatch
    acc_float: float             # test accuracy of the float network the run starts from

    @property
    def quantizer(self):
        """The watched layer's quantizer."""
        qmin, qmax = qrange(self.bits)
        s = np.full((1, 1), self.scales[WATCH])
        return Quantizer(self.bits, s, np.zeros_like(s), qmin, qmax)

    def snapshot(self, step):
        """Index of the last recorded step at or before ``step``."""
        return int(np.searchsorted(self.steps, step, side="right") - 1)


def _snapshot_steps(steps, dense=10, count=24):
    """Every step up to ``dense``, then ``count`` log-spaced steps to the end."""
    log = np.round(np.logspace(np.log10(dense + 1), np.log10(steps), count)).astype(int)
    return np.unique(np.concatenate([np.arange(dense + 1), log, [steps]]))


def qat_train(bits, epochs=QAT_EPOCHS, lr=QAT_LR, batch=QAT_BATCH, seed=0, cache_dir=None):
    """Fine-tune the trained MLP with fake-quantized weights (per-tensor
    symmetric scales, calibrated once by :func:`calibrate_mse` and fixed) and
    the straight-through estimator, by Adam; cached under its settings."""
    settings = dict(model="mlp_qat", bits=bits, epochs=epochs, lr=lr, batch=batch, seed=seed,
                    scheme="per-tensor symmetric, mse-calibrated, fixed", ste="clipped",
                    watch=WATCH, start="mnist.train() defaults")

    def build():
        params, acc_float, (xtr, ytr, xte, yte) = trained_mlp(cache_dir)
        qs = [calibrate_mse(W, bits)[0] for W, _ in params]
        P = {}
        for k, (W, b) in enumerate(params):
            P["W%d" % k], P["b%d" % k] = W.copy(), b.copy()
        rng = np.random.default_rng(seed)
        opt = Adam(P, lr)
        eye = np.eye(CLASSES)
        n_steps = epochs * (-(-len(xtr) // batch))
        marks = _snapshot_steps(n_steps)
        want = set(marks.tolist())
        Ws, accs, losses, grad_q = [], [], [], None

        def record():
            Ws.append(P["W%d" % WATCH].copy())
            accs.append(accuracy([[qs[k].fake(P["W%d" % k]), P["b%d" % k]]
                                  for k in range(len(qs))], xte, yte))

        record()                                   # step 0: the PTQ network
        for step, idx in enumerate(minibatches(rng, len(xtr), epochs, batch), 1):
            y = eye[ytr[idx]]
            g, gq, p = qat_forward_backward(P, qs, xtr[idx], y)
            if grad_q is None:
                grad_q = gq[WATCH]
            losses.append(float(-np.mean(np.log(np.sum(p * y, 1) + 1e-300))))
            opt.step(P, g)
            if step in want:
                record()
        return dict(scales=np.array([float(q.scale.ravel()[0]) for q in qs]), steps=marks,
                    W=np.array(Ws), acc=np.array(accs), batch_loss=np.array(losses),
                    grad_q=grad_q, acc_float=np.array(acc_float))

    d = cached_npz("quant_qat%d" % bits, settings, build, cache_dir)
    return QATRun(bits, d["scales"], d["steps"], d["W"], d["acc"], d["batch_loss"], d["grad_q"],
                  float(d["acc_float"]))


# ------------------------------------------------------------------ number formats


def round_bf16(x):
    """``x`` rounded to the nearest bfloat16 (round half to even), returned
    as float32: the upper 16 bits of the float32 pattern are kept, the lower
    16 rounded away.  Infinities and NaNs pass through; a finite value too
    large for the format becomes infinity, as a real cast would."""
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    nan = np.isnan(x)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    return np.where(nan, x, r).astype(np.float32)


def round_to(x, fmt):
    """``x`` rounded to the format and returned as float32 (float16 through
    numpy's own type, bfloat16 through :func:`round_bf16`); overflow gives
    infinity silently, as a cast does."""
    if fmt == "float32":
        return np.asarray(x, np.float32)
    if fmt == "float16":
        with np.errstate(over="ignore"):
            return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)
    if fmt == "bfloat16":
        return round_bf16(x)
    raise ValueError(fmt)


def format_info(fmt):
    """Field widths and limits of a format: ``dict(sign, exponent, mantissa,
    bits, max, tiny, smallest_subnormal)``; the float32 numbers are
    numpy's, bfloat16's follow from its 8-bit exponent and 7-bit mantissa."""
    s, e, m = FIELDS[fmt]
    if fmt == "bfloat16":
        f32 = np.finfo(np.float32)
        largest = float(np.uint32(0x7F7F0000).view(np.float32))    # all exponent bits but one
        return dict(sign=s, exponent=e, mantissa=m, bits=s + e + m, max=largest,
                    tiny=float(f32.tiny), smallest_subnormal=float(f32.tiny) * 2.0 ** -m)
    fi = np.finfo(np.float16 if fmt == "float16" else np.float32)
    return dict(sign=s, exponent=e, mantissa=m, bits=s + e + m, max=float(fi.max),
                tiny=float(fi.tiny), smallest_subnormal=float(fi.smallest_subnormal))


def bit_fields(value, fmt):
    """``(sign, exponent, mantissa)`` as strings of '0'/'1': the bit pattern
    of ``value`` rounded to ``fmt``."""
    s, e, m = FIELDS[fmt]
    if fmt == "float16":
        u = int(np.float16(value).view(np.uint16))
    else:
        u = int(round_to(value, fmt).reshape(()).view(np.uint32))
        if fmt == "bfloat16":
            u >>= 16
    bits = format(u, "0%db" % (s + e + m))
    return bits[:s], bits[s:s + e], bits[s + e:]


def next_up(value, fmt):
    """The smallest representable number of ``fmt`` above ``value`` (a
    representable value itself)."""
    if fmt == "float16":
        return float(np.nextafter(np.float16(value), np.float16(np.inf)))
    if fmt == "float32":
        return float(np.nextafter(np.float32(value), np.float32(np.inf)))
    u = round_to(value, fmt).reshape(()).view(np.uint32)
    step = np.uint32(0x00010000)
    if value >= 0:
        return float(np.uint32(u + step).view(np.float32))
    if u == np.uint32(0x80000000):              # -0: the next value up is the smallest positive
        return float(np.uint32(step).view(np.float32))
    return float(np.uint32(u - step).view(np.float32))


def ulp(value, fmt):
    """Spacing of the representable numbers of ``fmt`` just above ``value``."""
    v = float(round_to(value, fmt))
    return next_up(v, fmt) - v


def representable(lo, hi, fmt, limit=400):
    """The format's representable values in ``[lo, hi]`` ascending, or None
    when there are more than ``limit`` (a grid too fine to draw)."""
    v = float(round_to(lo, fmt))
    if v < lo:
        v = next_up(v, fmt)
    out = []
    while v <= hi:
        out.append(v)
        if len(out) > limit:
            return None
        v = next_up(v, fmt)
    return np.array(out)


def classify(g, fmt):
    """Of the non-zero values of ``g``, the fraction that cast to ``fmt`` as
    0 (underflow), as a subnormal (reduced precision) and as a normal number:
    ``(zero, subnormal, normal)``."""
    g = np.asarray(g, np.float64)
    nz = g != 0
    r = round_to(g, fmt).astype(np.float64)
    info = format_info(fmt)
    zero = (r == 0) & nz
    sub = (np.abs(r) < info["tiny"]) & (r != 0)
    return float(zero.sum() / nz.sum()), float(sub.sum() / nz.sum()), float(
        ((np.abs(r) >= info["tiny"]) & nz).sum() / nz.sum())


def loss_scaled(g, scale, fmt):
    """``g`` as it survives a cast to ``fmt`` with loss scaling: multiplied by
    ``scale``, rounded, divided back in float32 (so the result is float32
    with ``fmt``'s precision)."""
    return round_to(np.asarray(g, np.float64) * scale, fmt).astype(np.float64) / scale


# ------------------------------------------------------------------ the underflow example


def underflow_gradient(cache_dir=None):
    """The weight gradient of hidden layer :data:`DEEP_LAYER` of
    :data:`DEEP_SPEC` at initialisation (seed 0), on the first
    :data:`DEEP_BATCH` images of the initialisation animation's training
    subset: a real ``(100, 100)`` gradient whose values sit around float16's
    smallest subnormal.  Cached; returns ``dict(g, rms)``."""
    settings = dict(model="deepnet_grad", spec=DEEP_SPEC.settings(), layer=DEEP_LAYER,
                    batch=DEEP_BATCH, seed=0, data="init_anim subset 10000/2000 seed 0")

    def build():
        data = subset(10000, 2000, seed=0, cache_dir=cache_dir or DEFAULT_CACHE)
        P = init(DEEP_SPEC, np.random.default_rng(0))
        x, y = data.x_train[:DEEP_BATCH], data.onehot(data.y_train[:DEEP_BATCH])
        G, _D = backward(DEEP_SPEC, P, forward(DEEP_SPEC, P, x), y)
        g = G["W%d" % DEEP_LAYER]
        return dict(g=g, rms=np.array(np.sqrt(np.mean(g ** 2))))

    d = cached_npz("quant_underflow", settings, build, cache_dir)
    return dict(g=d["g"], rms=float(d["rms"]))


# ------------------------------------------------------------------ training in a format


@dataclass
class PrecisionRun:
    """One recorded run of :func:`precision_train`."""

    fmt: str
    master: bool
    loss_scale: float
    batch_loss: np.ndarray       # (steps,) minibatch cross-entropy before each update (NaN after
                                 # the run diverged)
    eval_steps: np.ndarray       # (S,) steps after which the test accuracy was measured
    test_acc: np.ndarray         # (S,)

    @property
    def steps(self):
        """Number of updates."""
        return len(self.batch_loss)

    @property
    def diverged(self):
        """The first step whose loss was not finite, or -1."""
        bad = np.nonzero(~np.isfinite(self.batch_loss))[0]
        return int(bad[0]) if len(bad) else -1

    @property
    def acc(self):
        """Test accuracy at the end."""
        return float(self.test_acc[-1])


def _adam_step(P, g, m, v, t, lr, store):
    """Adam's update with every stored array rounded by ``store`` (the
    optimiser's own format: the training format, or float32 under the
    master-weights pattern); the arithmetic is :class:`voxanim.optim.Adam`'s."""
    for k in P:
        m[k] = store(BETA1 * m[k] + GAIN1 * g[k])
        v[k] = store(BETA2 * v[k] + GAIN2 * g[k] ** 2)
        step = lr * (m[k] / (1 - BETA1 ** t)) / (np.sqrt(v[k] / (1 - BETA2 ** t)) + EPS)
        P[k] = store(P[k] - step)


def precision_step(P, x, y_onehot, r, loss_scale=1.0):
    """One forward and backward pass of the MLP with every tensor rounded
    by ``r`` (the format) and float32 accumulation.

    Returns ``(grads, loss)``: the gradients of ``loss_scale`` times the mean
    cross-entropy, rounded, still scaled; ``loss`` the unscaled float loss.
    """
    n = len(P) // 2
    with np.errstate(invalid="ignore", over="ignore"):      # a diverged run computes with inf / NaN
        W = [r(P["W%d" % k]) for k in range(n)]
        b = [r(P["b%d" % k]) for k in range(n)]
        h = r(x)
        acts, pre = [h], []
        for k in range(n):
            z = r(h @ W[k] + b[k])
            pre.append(z)
            if k < n - 1:
                h = np.maximum(z, 0.0)
                acts.append(h)
        e = np.exp(pre[-1] - pre[-1].max(1, keepdims=True))
        p = r(e / e.sum(1, keepdims=True))
        loss = float(-np.mean(np.log(np.sum(p.astype(np.float64) * y_onehot, 1) + 1e-300)))
        delta = r((p - y_onehot) * (loss_scale / len(x)))
        g = {}
        for k in reversed(range(n)):
            g["W%d" % k] = r(acts[k].T @ delta)
            g["b%d" % k] = r(delta.sum(0))
            if k > 0:
                delta = r((delta @ W[k].T) * (pre[k - 1] > 0))
    return g, loss


def precision_train(fmt, master=False, loss_scale=1.0, epochs=PRECISION_EPOCHS, lr=PRECISION_LR,
                    batch=PRECISION_BATCH, seed=0, cache_dir=None):
    """Train the MLP from its usual He-normal start with every tensor in
    ``fmt`` (see the module docstring), by Adam, and record it; cached.

    With ``master`` the parameters and Adam's moments stay in float32 and
    only the copies used by the forward and backward passes are rounded;
    ``loss_scale`` multiplies the loss before the backward pass and divides
    the gradients after it (in float32; meaningful with master weights).
    """
    settings = dict(model="mlp_precision", fmt=fmt, master=master, loss_scale=loss_scale,
                    epochs=epochs, lr=lr, batch=batch, seed=seed, init="he_normal",
                    optimizer="adam", accumulate="float32", data="mnist/255")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        def r(a):
            return round_to(a, fmt)

        def store(a):
            return np.asarray(a, np.float32) if master else r(a)
        P = {"%s%d" % (name, k): store(a) for k, layer in enumerate(_init(rng))
             for name, a in zip("Wb", layer, strict=True)}
        m = {k: np.zeros_like(a) for k, a in P.items()}
        v = {k: np.zeros_like(a) for k, a in P.items()}
        eye = np.eye(CLASSES, dtype=np.float32)
        losses, eval_steps, accs = [], [], []
        for t, idx in enumerate(minibatches(rng, len(xtr), epochs, batch), 1):
            g, loss = precision_step(P, xtr[idx].astype(np.float32), eye[ytr[idx]], r, loss_scale)
            losses.append(loss)
            if loss_scale != 1.0:
                g = {k: a.astype(np.float32) / np.float32(loss_scale) for k, a in g.items()}
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                _adam_step(P, g, m, v, t, lr, store)
            if t % EVAL_EVERY == 0:
                eval_steps.append(t)
                params = [[r(P["W%d" % k]), r(P["b%d" % k])] for k in range(len(P) // 2)]
                with np.errstate(invalid="ignore", over="ignore"):
                    accs.append(accuracy(params, xte.astype(np.float32), yte))
        return dict(batch_loss=np.array(losses), eval_steps=np.array(eval_steps),
                    test_acc=np.array(accs))

    tag = "quant_%s%s%s" % (fmt, "_master" if master else "",
                            "_x%d" % loss_scale if loss_scale != 1.0 else "")
    d = cached_npz(tag, settings, build, cache_dir)
    return PrecisionRun(fmt, master, loss_scale, d["batch_loss"], d["eval_steps"], d["test_acc"])


PRECISION_RUNS = (          # (label, format, master weights): the training phase's curves
    ("float32", "float32", False),
    ("float16", "float16", False),
    ("bfloat16", "bfloat16", False),
    ("float16 + float32 master weights", "float16", True),
)


def precision_runs(cache_dir=None):
    """The recorded runs of :data:`PRECISION_RUNS`, in order."""
    return [precision_train(fmt, master, cache_dir=cache_dir)
            for _label, fmt, master in PRECISION_RUNS]

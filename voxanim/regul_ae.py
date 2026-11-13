"""A regularised autoencoder for MNIST in plain numpy: one wide code layer,
trained plain and under four regularisers.

    x (784) -> h = ReLU(x W + be) (256, the code) -> r = sigmoid(h W^T + bd) (784)

The code is wider than the 2-D and 3-D bottlenecks of :mod:`voxanim.autoencoder`
and nothing forces it to be small, so what the code becomes is up to the
regulariser.  The weights are **tied**: the decoder is the encoder's
transpose, so each code unit has one 28 x 28 feature, read in by the encoder
and written out by the decoder.  Without the tie, a penalty on the code (L1,
contractive) is dodged by scale: the encoder shrinks and the decoder grows to
match, and the code is no sparser or flatter than before (measured: the
untied contractive encoder fell to 1/30 of its norm, the decoder doubled).
The objective is the mean squared reconstruction error over the batch and
the pixels, plus one of

* ``"sparse"`` -- an **L1 penalty on the code**, ``alpha * mean_n sum_j h_nj``
  (``h >= 0``, so that is its L1 norm): most units are exactly 0 for any one
  digit,
* ``"decay"`` -- **L2 weight decay**, ``lam / 2 * |W|^2``,
* ``"denoise"`` -- **denoising**: the encoder sees the digit plus Gaussian
  noise of std ``sigma`` (fresh every step, zero mean, so the expected input
  is the clean digit) and must rebuild the clean digit,
* ``"contract"`` -- the **contractive penalty**, ``gamma * mean_n |J_n|_F^2``
  with ``J_n = dh / dx`` at digit ``n``.  For a ReLU code ``J_n`` is ``W``'s
  columns of the units that are on, so ``|J_n|_F^2 = sum_j [z_nj > 0]
  |W_:j|^2``: each unit's weights are decayed in proportion to how often it
  is on (its derivative through the on/off switch is 0 almost everywhere).

:func:`forward`, :func:`objective` and :func:`backward` are checked against
finite differences in ``tests/test_regul.py``.  :func:`train` (Adam,
:class:`voxanim.optim.Adam`) records the test error at a geometric schedule
of steps and the weights at a few of them; :func:`measure` reads off what
the animation shows: reconstructions of clean and corrupted test digits, how
many code units one digit uses, how far the code moves when the digit is
corrupted, and the Jacobian's size.  Everything is seeded and cached by
:func:`voxanim.cache.cached_npz`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .autoencoder import sigmoid, snapshot_steps
from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import minibatches
from .optim import Adam

INPUTS = 784
CODE = 256                        # code units (16 x 16)
KINDS = ("plain", "sparse", "decay", "denoise", "contract")


@dataclass(frozen=True)
class AESpec:
    """One autoencoder: its regulariser and that regulariser's strength."""

    kind: str = "plain"           # one of KINDS
    strength: float = 0.0         # alpha, lam, sigma or gamma
    code: int = CODE

    def settings(self):
        """The JSON-able dict that keys a cache file."""
        return asdict(self)


def init(spec, rng):
    """He-initialised tied weights ``W`` ``(784, code)``, zero biases."""
    return dict(W=rng.normal(0.0, np.sqrt(2.0 / INPUTS), (INPUTS, spec.code)),
                be=np.zeros(spec.code), bd=np.zeros(INPUTS))


def encode(P, x):
    """``(z, h)``: the code's pre-activations and the code of ``x`` ``(N, 784)``."""
    z = x @ P["W"] + P["be"]
    return z, np.maximum(z, 0.0)


def decode(P, h):
    """The reconstruction of code ``h``: each unit adds its feature (a column
    of ``W``) in proportion to its value."""
    return sigmoid(h @ P["W"].T + P["bd"])


def forward(P, x_in):
    """The batch ``x_in`` through the autoencoder: ``dict(x_in, z, h, r)``."""
    z, h = encode(P, x_in)
    return dict(x_in=x_in, z=z, h=h, r=decode(P, h))


def recon_error(r, x):
    """Mean squared error over the batch and the pixels."""
    return float(np.mean((r - x) ** 2))


def jacobian_sq(P, z):
    """``(N,)`` ``|dh/dx|_F^2`` of a ReLU code at pre-activations ``z``."""
    col = (P["W"] ** 2).sum(0)
    return (z > 0) @ col


def penalty(spec, P, f):
    """The regulariser's term of the objective for the forward pass ``f``."""
    s = spec.strength
    if spec.kind == "sparse":
        return float(s * f["h"].sum(1).mean())
    if spec.kind == "decay":
        return float(0.5 * s * (P["W"] ** 2).sum())
    if spec.kind == "contract":
        return float(s * jacobian_sq(P, f["z"]).mean())
    return 0.0


def objective(spec, P, f, x):
    """Reconstruction error against the clean ``x`` plus the penalty."""
    return recon_error(f["r"], x) + penalty(spec, P, f)


def backward(spec, P, f, x):
    """Gradients of :func:`objective` with respect to every parameter."""
    n = len(x)
    r = f["r"]
    dr = 2.0 * (r - x) / r.size
    du = dr * r * (1.0 - r)
    dh = du @ P["W"]
    if spec.kind == "sparse":
        dh = dh + spec.strength / n
    dz = dh * (f["z"] > 0)
    G = dict(W=f["x_in"].T @ dz + du.T @ f["h"], be=dz.sum(0), bd=du.sum(0))
    if spec.kind == "decay":
        G["W"] = G["W"] + spec.strength * P["W"]
    elif spec.kind == "contract":
        on = (f["z"] > 0).mean(0)                       # how often each unit is on
        G["W"] = G["W"] + 2.0 * spec.strength * P["W"] * on[None, :]
    return G


def corrupt(rng, x, sd):
    """``x`` plus Gaussian noise of std ``sd`` on every pixel."""
    return x + rng.normal(0.0, sd, x.shape)


# ------------------------------------------------------------------ training

EPOCHS, BATCH, LR, SEED = 10, 128, 1e-3, 0
EVAL_N = 2000                     # test digits a recorded error is measured on
SNAP_RATIO = 1.25                 # the recorded steps: geometric ...
WEIGHT_STEPS = (0, 15, 40, 100, 250, 600, 1500)   # ... plus these, which also keep the weights


@dataclass
class AERun:
    """One recorded training run (``S`` recorded steps)."""

    spec: AESpec
    steps: np.ndarray             # (S,)
    test_error: np.ndarray        # (S,) clean reconstruction error on EVAL_N test digits
    active: np.ndarray            # (S,) mean fraction of code units on per test digit
    snap_steps: np.ndarray        # (K,) the steps whose weights were kept
    W_snaps: np.ndarray           # (K, 784, code) float32
    P: dict                       # the trained parameters


def train(spec, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED, cache_dir=DEFAULT_CACHE) -> AERun:
    """Train ``spec`` on the 60,000 MNIST training digits with Adam (cached)."""
    from .mnist import load
    settings = dict(spec.settings(), weights="tied", noise="gaussian", epochs=epochs,
                    batch=batch, lr=lr, seed=seed,
                    eval_n=EVAL_N, snap_ratio=SNAP_RATIO, weight_steps=list(WEIGHT_STEPS))

    def build():
        xtr, _ytr, xte, _yte = load(cache_dir)
        xev = xte[:EVAL_N]
        rng = np.random.default_rng(seed)
        nrng = np.random.default_rng([seed, 1])          # the denoiser's noise
        P = init(spec, rng)
        opt = Adam(P, lr)
        total = epochs * (-(-len(xtr) // batch))
        keep = {s for s in WEIGHT_STEPS if s < total} | {total}
        marks = np.union1d(snapshot_steps(total, SNAP_RATIO), sorted(keep))
        rec = dict(test_error=[], active=[])
        snaps, snap_at = [], []
        want = set(marks.tolist())
        for step, idx in enumerate(minibatches(rng, len(xtr), epochs, batch)):
            if step in want:
                f = forward(P, xev)
                rec["test_error"].append(recon_error(f["r"], xev))
                rec["active"].append(float((f["h"] > 0).mean()))
                if step in keep:
                    snaps.append(P["W"].astype(np.float32))
                    snap_at.append(step)
            if step >= total:
                break
            x = xtr[idx]
            x_in = corrupt(nrng, x, spec.strength) if spec.kind == "denoise" else x
            f = forward(P, x_in)
            opt.step(P, backward(spec, P, f, x))
        if total in want and len(rec["test_error"]) < len(marks):
            f = forward(P, xev)
            rec["test_error"].append(recon_error(f["r"], xev))
            rec["active"].append(float((f["h"] > 0).mean()))
            snaps.append(P["W"].astype(np.float32))
            snap_at.append(total)
        return dict(P, steps=marks, test_error=np.array(rec["test_error"]),
                    active=np.array(rec["active"]), snap_steps=np.array(snap_at),
                    W_snaps=np.array(snaps, np.float32))

    d = cached_npz("regul_ae", settings, build, cache_dir)
    P = {k: d[k] for k in ("W", "be", "bd")}
    return AERun(spec, d["steps"], d["test_error"], d["active"], d["snap_steps"], d["W_snaps"], P)


# ------------------------------------------------------------------ measurements

NOISE_SD = 0.4                    # the corrupted test digits: Gaussian noise of this std
N_MEASURE = 2000                  # test digits the measurements average over


def measure(P, x, sd=NOISE_SD, seed=SEED):
    """What one trained autoencoder does to the test digits ``x``:

    ``error`` (clean in, clean out), ``noisy_error`` (noise of std ``sd`` in,
    compared with the clean digit), ``active`` (mean fraction of
    code units on per digit), ``shift`` (how far the code moves when the digit
    is corrupted, relative to the code's own size, ``|h(x~) - h(x)| / |h(x)|``),
    ``jac`` (mean ``|dh/dx|_F``) and ``w_norm`` (``|W|``)."""
    rng = np.random.default_rng([seed, 2])
    xn = corrupt(rng, x, sd)
    f, fn = forward(P, x), forward(P, xn)
    hn = np.linalg.norm(f["h"], axis=1)
    shift = np.linalg.norm(fn["h"] - f["h"], axis=1) / np.maximum(hn, 1e-12)
    return dict(error=recon_error(f["r"], x), noisy_error=recon_error(fn["r"], x),
                active=float((f["h"] > 0).mean()), shift=float(np.median(shift)),
                jac=float(np.sqrt(jacobian_sq(P, f["z"])).mean()),
                w_norm=float(np.linalg.norm(P["W"])))

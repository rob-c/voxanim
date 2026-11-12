"""A small denoising diffusion model (DDPM) on MNIST in plain numpy.

The **forward process** adds Gaussian noise to an image step by step, a
random walk in pixel space over :data:`T_STEPS` steps:

    x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z_t,     z_t ~ N(0, I)

which, with alpha_t = 1 - beta_t and abar_t = prod alpha_s, is the same as
jumping straight from the clean image:

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,         eps ~ N(0, I)

The schedule (:func:`schedule`) is the cosine schedule of Nichol & Dhariwal
(2021): abar_t = cos^2((t / T + s) / (1 + s) * pi / 2), which spends more
steps at low noise than the linear one and ends at abar_T ~ 0, pure noise.

The **denoiser** (:func:`forward`) is a dense network that reads the noisy
image x_t and the step t and predicts the noise eps that was added:

    h1 = silu(x_t W0 + b0 + emb(t) Wt + bt)      emb: sinusoidal time embedding
    h2 = silu(h1 W1 + b1)
    eps^ = g(t) x_t + h2 W2 + b2                 784 -> 512 -> 512 -> 784, plus a skip
    g(t) = emb(t) Wg + bg                        the skip's gain, a scalar per step

The skip connection matters: white noise has no structure, so a network
whose hidden layers are narrower than the image cannot pass it through (the
best a 512-wide bottleneck can do on eps alone is to lose 272 of 784
dimensions, a loss of 0.35 even at t = T where eps is simply x_t).  With
the skip, the gain carries x_t straight to the output and the hidden layers
only have to subtract the digit: the one-dimensional thing in x_t.  This is
the role the skip connections of a U-Net play.  (SiLU, z sigmoid(z), is the
activation diffusion networks use: with ReLU and Adam at this learning rate a
quarter of the units die in the first few hundred steps.  The output layer
starts small, :data:`OUT_SCALE`, and the gain at 1.)

trained on the mean squared error ``|eps^ - eps|^2`` (:func:`loss_grads`),
each image of the batch at its own random step.  Pixels are scaled to
[-1, 1] (:func:`to_signal`) so the signal and the noise have the same scale.

The **reverse process** (:func:`reverse_step`, :func:`sample`) starts from
pure noise x_T and walks back: at every step the denoiser's eps^ gives an
estimate of the clean image,

    x0^ = (x_t - sqrt(1 - abar_t) eps^) / sqrt(abar_t),     clipped to [-1, 1]

and the next sample is drawn from the posterior q(x_{t-1} | x_t, x0^):

    mean  = sqrt(abar_{t-1}) beta_t / (1 - abar_t) x0^
            + sqrt(alpha_t) (1 - abar_{t-1}) / (1 - abar_t) x_t
    var   = beta~_t = beta_t (1 - abar_{t-1}) / (1 - abar_t)

Clipping x0^ is what keeps a small model's samples on the manifold of
images; without it the two forms of the mean (via x0^ or via eps^) are the
same algebra, which :mod:`tests.test_diffusion` checks.

Training (:func:`train`) is seeded, uses :class:`voxanim.optim.Adam` and is
cached by :func:`voxanim.cache.cached_npz`; :func:`trace` records what the
animation draws: the forward walk of a few test digits, the reverse chain
from fixed-seed noise, one worked training example and the loss curve.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import minibatches
from .mnist import load
from .optim import Adam

T_STEPS = 200                    # diffusion steps
HIDDEN = 512                     # width of the two hidden layers
EMBED = 64                       # sinusoidal time embedding: EMBED / 2 frequencies
EMBED_BASE = 10000.0             # the embedding's longest period, in steps
COSINE_S = 0.008                 # the cosine schedule's offset (Nichol & Dhariwal)
BETA_MAX = 0.999                 # the schedule's beta_t is clipped here (the last steps)
LINEAR_BETA = (1e-4, 0.02)       # the linear schedule's first and last beta (Ho et al.)
OUT_SCALE = 0.1                  # the output layer's initial weights, relative to He
EPOCHS, BATCH, LR = 20, 512, 1e-3
WIDTHS = (784, HIDDEN, HIDDEN, 784)


# ------------------------------------------------------------------ the schedule


def schedule(T=T_STEPS, kind="cosine"):
    """``(beta, alpha, abar)``, each ``(T + 1,)`` indexed by the step ``t`` (entry
    0 is the clean image: beta_0 = 0, abar_0 = 1).

    ``kind`` is ``"cosine"`` (the default, see the module docstring) or
    ``"linear"`` (beta from :data:`LINEAR_BETA` [0] to [1] in equal steps).
    """
    t = np.arange(T + 1, dtype=np.float64)
    if kind == "cosine":
        f = np.cos((t / T + COSINE_S) / (1 + COSINE_S) * np.pi / 2) ** 2
        beta = np.minimum(1.0 - f[1:] / f[:-1], BETA_MAX)
    elif kind == "linear":
        beta = np.linspace(LINEAR_BETA[0], LINEAR_BETA[1], T)
    else:
        raise ValueError("unknown schedule %r" % kind)
    beta = np.concatenate([[0.0], beta])
    alpha = 1.0 - beta
    return beta, alpha, np.cumprod(alpha)


def to_signal(x):
    """Pixels in [0, 1] -> the diffusion's [-1, 1]."""
    return 2.0 * np.asarray(x, np.float64) - 1.0


def to_image(s):
    """The diffusion's [-1, 1] -> pixels in [0, 1] (clipped)."""
    return np.clip((np.asarray(s, np.float64) + 1.0) / 2.0, 0.0, 1.0)


def noised(x0, t, eps, abar):
    """``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`` for steps ``t`` ``(N,)``
    and ``x0, eps`` ``(N, 784)``."""
    a = abar[t][:, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps


def forward_walk(x0, rng, beta):
    """The forward process step by step: ``(T + 1, N, 784)``, row ``t`` is ``x_t``.

    Fresh noise at every step: ``x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z_t``.
    """
    xs = [np.asarray(x0, np.float64)]
    for t in range(1, len(beta)):
        z = rng.standard_normal(xs[-1].shape)
        xs.append(np.sqrt(1.0 - beta[t]) * xs[-1] + np.sqrt(beta[t]) * z)
    return np.array(xs)


# ------------------------------------------------------------------ the denoiser


def time_embedding(t, dim=EMBED, base=EMBED_BASE):
    """Sinusoidal embedding ``(N, dim)`` of integer steps ``t`` ``(N,)``:
    ``sin(t / base^(2k / dim))`` and ``cos`` of the same, ``k = 0 .. dim / 2 - 1``."""
    t = np.asarray(t, np.float64)[:, None]
    k = np.arange(dim // 2, dtype=np.float64)
    freq = 1.0 / base ** (2.0 * k / dim)
    ang = t * freq[None]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def init(rng, widths=WIDTHS, embed=EMBED):
    """He-initialised weights ``W0, W1`` and a small ``W2`` (:data:`OUT_SCALE`),
    zero biases, and the time embedding's ``Wt, bt`` into the first hidden layer."""
    P = {}
    for k, (a, b) in enumerate(itertools.pairwise(widths)):
        P["W%d" % k] = rng.normal(0.0, np.sqrt(2.0 / a), (a, b))
        P["b%d" % k] = np.zeros(b)
    P["W2"] *= OUT_SCALE
    P["Wt"] = rng.normal(0.0, np.sqrt(2.0 / embed), (embed, widths[1]))
    P["bt"] = np.zeros(widths[1])
    P["Wg"], P["bg"] = np.zeros((embed, 1)), np.ones(1)      # the skip's gain starts at 1
    return P


def sigmoid(z):
    """Logistic sigmoid (``z`` clipped so ``exp`` cannot overflow)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def silu(z):
    """``z * sigmoid(z)``."""
    return z * sigmoid(z)


def silu_grad(z):
    """``d silu / dz = sigmoid(z) (1 + z (1 - sigmoid(z)))``."""
    s = sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def forward(P, x, t):
    """The denoiser on ``x`` ``(N, 784)`` at steps ``t`` ``(N,)``.

    Returns ``emb`` (the time embedding), ``z1, h1, z2, h2`` (pre-activations
    and SiLU outputs of the hidden layers), ``g`` (the skip's gain, ``(N, 1)``)
    and ``eps`` (the predicted noise).
    """
    emb = time_embedding(t)
    z1 = x @ P["W0"] + P["b0"] + emb @ P["Wt"] + P["bt"]
    h1 = silu(z1)
    z2 = h1 @ P["W1"] + P["b1"]
    h2 = silu(z2)
    g = emb @ P["Wg"] + P["bg"]
    return dict(x=x, emb=emb, z1=z1, h1=h1, z2=z2, h2=h2, g=g,
                eps=g * x + h2 @ P["W2"] + P["b2"])


def loss_grads(P, f, eps):
    """``(loss, grads)``: the mean squared error ``mean((eps^ - eps)^2)`` of the
    forward pass ``f`` against the true noise ``eps``, and its gradients."""
    d = 2.0 * (f["eps"] - eps) / eps.size
    g = {"W2": f["h2"].T @ d, "b2": d.sum(0)}
    dg = (d * f["x"]).sum(1, keepdims=True)                    # through the skip's gain
    g["Wg"], g["bg"] = f["emb"].T @ dg, dg.sum(0)
    d = (d @ P["W2"].T) * silu_grad(f["z2"])
    g["W1"], g["b1"] = f["h1"].T @ d, d.sum(0)
    d = (d @ P["W1"].T) * silu_grad(f["z1"])
    g["W0"], g["b0"] = f["x"].T @ d, d.sum(0)
    g["Wt"], g["bt"] = f["emb"].T @ d, d.sum(0)
    return float(np.mean((f["eps"] - eps) ** 2)), g


# ------------------------------------------------------------------ sampling


def predict_x0(x_t, eps_hat, t, abar, clip=True):
    """The clean image implied by the noise estimate:
    ``(x_t - sqrt(1 - abar_t) eps^) / sqrt(abar_t)``, clipped to [-1, 1]."""
    a = abar[t]
    x0 = (x_t - np.sqrt(1.0 - a) * eps_hat) / np.sqrt(a)
    return np.clip(x0, -1.0, 1.0) if clip else x0


def posterior(x_t, x0, t, beta, alpha, abar):
    """Mean and variance of ``q(x_{t-1} | x_t, x_0)`` (DDPM eq. 6 and 7)."""
    ab, ab_prev = abar[t], abar[t - 1]
    mean = (np.sqrt(ab_prev) * beta[t] / (1.0 - ab) * x0
            + np.sqrt(alpha[t]) * (1.0 - ab_prev) / (1.0 - ab) * x_t)
    var = beta[t] * (1.0 - ab_prev) / (1.0 - ab)
    return mean, var


def reverse_step(P, x_t, t, z, sched, clip=True):
    """One reverse step ``x_t -> x_{t-1}`` with the noise ``z`` (zeros at t = 1).

    Returns ``(x_{t-1}, x0^, eps^)``.
    """
    beta, alpha, abar = sched
    n = len(x_t)
    eps_hat = forward(P, x_t, np.full(n, t))["eps"]
    x0 = predict_x0(x_t, eps_hat, t, abar, clip)
    mean, var = posterior(x_t, x0, t, beta, alpha, abar)
    return mean + np.sqrt(var) * z, x0, eps_hat


def sample(P, rng, n, sched, record=False):
    """``n`` samples from pure noise; with ``record``, also the whole chain:
    ``xs`` ``(T + 1, n, 784)`` (``xs[t]`` is ``x_t``) and the ``x0`` estimates
    ``(T + 1, n, 784)`` (``x0s[t]`` is the estimate made at step ``t``; row 0 is
    the final sample).  No noise is added at the last step (t = 1)."""
    T = len(sched[0]) - 1
    x = rng.standard_normal((n, 784))
    xs, x0s = [None] * (T + 1), [None] * (T + 1)
    xs[T] = x
    for t in range(T, 0, -1):
        z = rng.standard_normal(x.shape) if t > 1 else np.zeros_like(x)
        x, x0, _eps = reverse_step(P, x, t, z, sched)
        xs[t - 1], x0s[t] = x, x0
    x0s[0] = x
    if record:
        return np.array(xs), np.array(x0s)
    return x


# ------------------------------------------------------------------ training


def train_settings(epochs=EPOCHS, batch=BATCH, lr=LR, seed=0, kind="cosine"):
    """Every setting that determines :func:`train`'s weights (its cache key)."""
    return dict(T=T_STEPS, widths=WIDTHS, embed=EMBED, embed_base=EMBED_BASE,
                activation="silu", skip="gain", out_scale=OUT_SCALE, schedule=kind,
                cosine_s=COSINE_S,
                epochs=epochs, batch=batch, lr=lr, seed=seed)


def train(epochs=EPOCHS, batch=BATCH, lr=LR, seed=0, kind="cosine", cache_dir=DEFAULT_CACHE):
    """Train the denoiser with Adam (or load the cache).

    Returns ``(P, losses)``: the parameters and the training loss of every
    step ``(steps,)``.  Each step draws a batch of training images, a random
    step ``t`` in 1..T for each and standard-normal noise, in that order from
    one seeded generator.
    """
    settings = train_settings(epochs, batch, lr, seed, kind)
    T = T_STEPS

    def build():
        xtr, _ytr, _xte, _yte = load(cache_dir)
        x_all = to_signal(xtr)
        _beta, _alpha, abar = schedule(T, kind)
        rng = np.random.default_rng(seed)
        P = init(rng)
        opt = Adam(P, lr)
        losses = []
        for idx in minibatches(rng, len(x_all), epochs, batch):
            x0 = x_all[idx]
            t = rng.integers(1, T + 1, len(idx))
            eps = rng.standard_normal(x0.shape)
            L, g = loss_grads(P, forward(P, noised(x0, t, eps, abar), t), eps)
            opt.step(P, g)
            losses.append(L)
        return dict(P, losses=np.array(losses, np.float32))

    d = cached_npz("diffusion", settings, build, cache_dir)
    return {k: v for k, v in d.items() if k != "losses"}, d["losses"]


# ------------------------------------------------------------------ the trace


@dataclass
class DiffusionTrace:
    """Everything the animation draws, from one trained denoiser.

    Images are in the diffusion's [-1, 1]; the step axis runs ``0 .. T``.
    """

    P: dict
    kind: str                     # the schedule
    beta: np.ndarray              # (T + 1,)
    alpha: np.ndarray             # (T + 1,)
    abar: np.ndarray              # (T + 1,)
    losses: np.ndarray            # (steps,) training loss per step
    steps_per_epoch: int
    digits: np.ndarray            # (k, 784) the test digits noised
    labels: np.ndarray            # (k,)
    walk: np.ndarray              # (T + 1, k, 784) their forward process, step by step
    example_t: int                # the worked training example's step ...
    example_x: np.ndarray         # (784,) its x_t
    example_eps: np.ndarray       # (784,) the noise added
    example_eps_hat: np.ndarray   # (784,) the denoiser's prediction
    example_loss: float           # mean (eps^ - eps)^2 on that one image
    chain: np.ndarray             # (T + 1, m, 784) reverse process, chain[t] = x_t
    chain_x0: np.ndarray          # (T + 1, m, 784) the x0^ estimate at each step
    test_loss: float              # the loss on TEST_N test images at random steps

    @property
    def T(self):
        """Number of diffusion steps."""
        return len(self.beta) - 1


WALK_SEED = 1                     # the forward walks' noise
EXAMPLE_SEED = 2                  # the worked example's noise
SAMPLE_SEED = 3                   # the reverse chains' starting noise
TEST_SEED = 4                     # the test loss's steps and noise
TEST_N = 2000                     # test images the test loss is measured on
DIGITS = (3, 2, 1, 18)            # test indices noised: a 0, a 1, a 2 and a 3
EXAMPLE_T = 60                    # the worked example's step


def trace(digits=DIGITS, samples=4, example_t=EXAMPLE_T, cache_dir=DEFAULT_CACHE):
    """Train (or load) the denoiser and record what the animation shows."""
    P, losses = train(cache_dir=cache_dir)
    kind = "cosine"
    sched = beta, _alpha, abar = schedule(T_STEPS, kind)
    _xtr, _ytr, xte, yte = load(cache_dir)
    x0 = to_signal(xte[list(digits)])
    walk = forward_walk(x0, np.random.default_rng(WALK_SEED), beta)
    eps = np.random.default_rng(EXAMPLE_SEED).standard_normal(784)
    xt = noised(x0[:1], np.array([example_t]), eps[None], abar)
    f = forward(P, xt, np.array([example_t]))
    ex_loss = float(np.mean((f["eps"][0] - eps) ** 2))
    chain, chain_x0 = sample(P, np.random.default_rng(SAMPLE_SEED), samples, sched, record=True)
    rng = np.random.default_rng(TEST_SEED)
    xs = to_signal(xte[:TEST_N])
    t = rng.integers(1, T_STEPS + 1, TEST_N)
    e = rng.standard_normal(xs.shape)
    test_loss = loss_grads(P, forward(P, noised(xs, t, e, abar), t), e)[0]
    per_epoch = -(-60000 // BATCH)
    return DiffusionTrace(P, kind, *sched, losses, per_epoch, x0, yte[list(digits)], walk,
                          example_t, xt[0], eps, f["eps"][0], ex_loss, chain, chain_x0,
                          test_loss)

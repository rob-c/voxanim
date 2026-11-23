"""The experiment behind animation 177, ``every-loss-is-a-likelihood``: a bridge from
fitting with chi-square and likelihoods (Part 1, statistics) to training a classifier
(Part 2, classical machine learning, and the networks after it), see
:func:`voxanim.next_catalog.redesigned`.

Three losses, each measured as minus the log of a likelihood on MNIST:

* **Gaussian errors: chi-square and squared error.** A measurement with errors: the ink
  of a digit's right half against the ink of its left half (each the mean pixel value of
  14 of the 28 columns), for 500 seeded training digits, binned by the left half's ink;
  each bin gives the mean right-half ink and its standard error. A straight line
  ``y = a + b x`` is fitted by minimising chi-square. For Gaussian errors,
  ``-2 ln L = chi^2 + sum ln(2 pi sigma_i^2)`` exactly, so the chi-square minimum and the
  maximum of the likelihood are the same slope (both curves over ``b``, with ``a`` at its
  best for each ``b``). The width of ``-ln L`` (its curvature) is the slope's error; it
  is checked against the scatter of the slope over 120 disjoint samples of 500 digits.
  With one sigma for every digit, chi-square is ``N / sigma^2`` times the mean squared
  error: least squares on the 500 digits is the maximum-likelihood line.
* **One of ten: cross-entropy.** A softmax regression (784 -> 10, one layer) trained
  with plain minibatch gradient descent on the 60,000 training digits; its cross-entropy
  on the 10,000 test digits is ``-(1/N) ln`` of the product of the probabilities it gave
  the right digits.
* **One of two: binary cross-entropy.** A logistic regression (784 -> 1) trained on the
  training 4s and 9s, scored on the test 4s and 9s: minus the log of a Bernoulli
  likelihood. The ten-class model read on the two logits of 4 and 9 alone is the same
  kind of model (a two-class softmax is a sigmoid of the logit difference).

Everything is plain numpy from seeded generators whose seeds are recorded. Training and
held-out digits are disjoint, and every loss and accuracy is a held-out one unless it
says otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
CLASSES = 10
EVIDENCE = dict(
    exact="exact arithmetic: chi-square, likelihoods and their logs worked out from the "
          "measured numbers",
    measured="measured on MNIST: fits and models made here in numpy, scored on held-out "
             "digits",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], mnist=mnist.SHA256, **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-loss-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== Gaussian errors
N_DIGITS = 500                     # digits in one sample
EDGES = np.linspace(0.02, 0.26, 11)  # ten bins of the left half's ink
MIN_BIN = 8                        # a bin with fewer digits is left out
GAUSS_SEED = 177
SAMPLES = 60000 // N_DIGITS        # disjoint samples: 120
GRID = 201                         # slopes on the scan, b-hat +- SPAN errors
SPAN = 4.0


def half_inks(images):
    """``(left, right)``: the mean pixel value of each image's left 14 columns and of its
    right 14."""
    im = np.asarray(images, float).reshape(-1, 28, 28)
    return im[:, :, :14].mean((1, 2)), im[:, :, 14:].mean((1, 2))


def bin_means(x, y, edges=EDGES, min_bin=MIN_BIN):
    """Digits binned by ``x``: each kept bin's mean ``x``, mean ``y``, the standard error
    of that mean (sample standard deviation / sqrt n) and its count. Digits outside the
    edges, and bins of fewer than ``min_bin``, are left out."""
    k = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        v = k == b
        n = int(v.sum())
        if n >= min_bin:
            rows.append((x[v].mean(), y[v].mean(), y[v].std(ddof=1) / np.sqrt(n), n))
    return np.array(rows).T if rows else np.zeros((4, 0))


def chi2(a, b, x, y, s):
    """``sum ((y - a - b x) / s)^2``; ``a`` and ``b`` may be arrays of trial lines."""
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    return (((y - a - b * x) / s) ** 2).sum(-1)


def gauss_loglik(a, b, x, y, s):
    """``ln L``: the sum of the logs of the Gaussian densities of each point about the
    line, written out from the density itself (not from chi-square)."""
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    dens = np.exp(-0.5 * ((y - a - b * x) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    return np.log(dens).sum(-1)


def weighted_line(x, y, s):
    """The line minimising chi-square: ``(a, b)`` and their covariance ``(A^T W A)^-1``."""
    w = 1.0 / np.asarray(s, float) ** 2
    A = np.stack([np.ones_like(x), x], 1)
    F = A.T @ (A * w[:, None])
    cov = np.linalg.inv(F)
    return cov @ (A.T @ (w * y)), cov


def best_a(b, x, y, s):
    """For each slope ``b``, the intercept that minimises chi-square."""
    w = 1.0 / np.asarray(s, float) ** 2
    b = np.asarray(b, float)
    return ((w * y).sum() - b * (w * x).sum()) / w.sum()


def mse(a, b, x, y):
    """The mean squared error of the line on every point; arrays of trial lines allowed."""
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    return ((y - a - b * x) ** 2).mean(-1)


def _gauss_build():
    x, _, xt, _ = mnist.load()
    L, R = half_inks(x)
    Lt, Rt = half_inks(xt)
    perm = np.random.default_rng(GAUSS_SEED).permutation(len(L))
    fits = []
    for j in range(SAMPLES):
        idx = perm[j * N_DIGITS:(j + 1) * N_DIGITS]
        bx, by, bs, bn = bin_means(L[idx], R[idx])
        p, cov = weighted_line(bx, by, bs)
        fits.append([p[0], p[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]),
                     float(chi2(p[0], p[1], bx, by, bs)), len(bx), bn.sum()])
    idx = perm[:N_DIGITS]
    bx, by, bs, bn = bin_means(L[idx], R[idx])
    p, cov = weighted_line(bx, by, bs)
    grid = p[1] + np.sqrt(cov[1, 1]) * np.linspace(-SPAN, SPAN, GRID)
    ag = best_a(grid, bx, by, bs)
    A = np.stack([np.ones(N_DIGITS), L[idx]], 1)
    q = np.linalg.lstsq(A, R[idx], rcond=None)[0]
    return dict(fits=np.array(fits), left=L[idx], right=R[idx], bins=np.array([bx, by, bs, bn]),
                line=p, cov=cov, grid=grid, grid_a=ag, chi2=chi2(ag, grid, bx, by, bs),
                loglik=gauss_loglik(ag, grid, bx, by, bs), lsq=q,
                mse_grid=mse(best_a(grid, L[idx], R[idx], np.ones(N_DIGITS)), grid, L[idx],
                             R[idx]),
                test_left=Lt, test_right=Rt)


@functools.lru_cache(maxsize=1)
def gauss_runs():
    """The line fits on 120 disjoint samples and the scans of the first (cached)."""
    return _cache("gauss", _gauss_build, (half_inks, bin_means, chi2, gauss_loglik,
                                          weighted_line, best_a, mse, _gauss_build),
                  n=N_DIGITS, edges=_floats(EDGES), min_bin=MIN_BIN, seed=GAUSS_SEED,
                  samples=SAMPLES, grid=GRID, span=SPAN)


def curvature(f, at, h):
    """``f''(at)`` by a central second difference with step ``h``."""
    return (f(at + h) - 2 * f(at) + f(at - h)) / h ** 2


# ====================================================================== one of ten, one of two
RATE = 0.5
BATCH = 100
EPOCHS = 5
SOFTMAX_SEED = 1770
LOGISTIC_SEED = 1771
EVERY = 60                         # softmax: steps between snapshots
LOG_EVERY = 13                     # logistic: steps between snapshots
PAIR = (4, 9)                      # the two classes: y = 1 for a 9
SHOWN_DIGIT = 0                    # the test digit followed


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def softmax_loss(W, c, x, y):
    """Mean cross-entropy (nats) of a softmax regression on a batch, and its gradients
    ``(dW, dc)``."""
    lp = log_softmax(x @ W + c)
    rows = np.arange(len(y))
    d = np.exp(lp)
    d[rows, y] -= 1.0
    d /= len(y)
    return float(-lp[rows, y].mean()), x.T @ d, d.sum(0)


def log_sigmoid(z):
    """``ln sigma(z)``, stable for large ``|z|``."""
    z = np.asarray(z, float)
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def binary_cross_entropy(z, t):
    """``-mean[t ln p + (1 - t) ln(1 - p)]`` with ``p = sigma(z)``: minus the log of a
    Bernoulli likelihood, per digit. ``1 - sigma(z) = sigma(-z)``."""
    return float(-(t * log_sigmoid(z) + (1 - t) * log_sigmoid(-z)).mean())


def logistic_loss(w, c, x, t):
    """Mean binary cross-entropy of a logistic regression on a batch, and its gradients."""
    z = x @ w + c
    d = (np.exp(log_sigmoid(z)) - t) / len(t)
    return binary_cross_entropy(z, t), x.T @ d, float(d.sum())


def pair_data():
    """``(x, t, xt, tt)``: the training and test 4s and 9s, ``t`` = 1 for a 9."""
    x, y, xt, yt = mnist.load()
    m, mt = np.isin(y, PAIR), np.isin(yt, PAIR)
    return x[m], (y[m] == PAIR[1]).astype(float), xt[mt], (yt[mt] == PAIR[1]).astype(float)


def _softmax_build():
    x, y, xt, yt = mnist.load()
    rng = np.random.default_rng(SOFTMAX_SEED)
    W, c = np.zeros((784, CLASSES)), np.zeros(CLASSES)
    per = len(x) // BATCH
    rows, step = [], 0

    def snap():
        lp, lpt = log_softmax(x @ W + c), log_softmax(xt @ W + c)
        rows.append([step, -lp[np.arange(len(y)), y].mean(), -lpt[np.arange(len(yt)), yt].mean(),
                     (lpt.argmax(1) == yt).mean()])
    snap()
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for b in range(per):
            idx = order[b * BATCH:(b + 1) * BATCH]
            _, gW, gc = softmax_loss(W, c, x[idx], y[idx])
            W -= RATE * gW
            c -= RATE * gc
            step += 1
            if step % EVERY == 0:
                snap()
    return dict(history=np.array(rows), W=W, c=c, test_logits=xt @ W + c)


@functools.lru_cache(maxsize=1)
def softmax_run():
    """The ten-class softmax regression, trained on 60,000 digits (cached)."""
    return _cache("softmax", _softmax_build, (log_softmax, softmax_loss, _softmax_build),
                  rate=RATE, batch=BATCH, epochs=EPOCHS, seed=SOFTMAX_SEED, every=EVERY)


def _logistic_build():
    x, t, xt, tt = pair_data()
    rng = np.random.default_rng(LOGISTIC_SEED)
    w, c = np.zeros(784), 0.0
    per = len(x) // BATCH
    rows, step = [], 0

    def snap():
        zt = xt @ w + c
        rows.append([step, binary_cross_entropy(x @ w + c, t), binary_cross_entropy(zt, tt),
                     ((zt > 0) == (tt > 0.5)).mean()])
    snap()
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for b in range(per):
            idx = order[b * BATCH:(b + 1) * BATCH]
            _, gw, gc = logistic_loss(w, c, x[idx], t[idx])
            w -= RATE * gw
            c -= RATE * gc
            step += 1
            if step % LOG_EVERY == 0:
                snap()
    return dict(history=np.array(rows), w=w, c=np.array(c), test_logits=xt @ w + c, test_t=tt)


@functools.lru_cache(maxsize=1)
def logistic_run():
    """The 4-against-9 logistic regression (cached)."""
    return _cache("logistic", _logistic_build, (log_sigmoid, binary_cross_entropy,
                                                logistic_loss, pair_data, _logistic_build),
                  rate=RATE, batch=BATCH, epochs=EPOCHS, seed=LOGISTIC_SEED,
                  every=LOG_EVERY, pair=list(PAIR))


# ====================================================================== the study
CORE = (log_softmax, log_sigmoid)


def every_loss_is_a_likelihood():
    g = gauss_runs()
    fits = g["fits"]
    bx, by, bs, bn = g["bins"]
    (a, b), cov = g["line"], g["cov"]
    sb = float(np.sqrt(cov[1, 1]))
    grid, c2, ll = g["grid"], g["chi2"], g["loglik"]
    const = -2 * ll - c2                                    # the same at every slope
    lo = int(np.argmin(c2))
    hi = int(np.argmax(ll))
    # the width of -ln L at its lowest point, with a at its best for each b
    nll = lambda v: -float(gauss_loglik(best_a(v, bx, by, bs), v, bx, by, bs))  # noqa: E731
    curv = curvature(nll, b, sb * 0.01)
    spread = float(fits[:, 1].std(ddof=1))
    # one sigma for every digit: least squares on the digits themselves
    L, R = g["left"], g["right"]
    qa, qb = g["lsq"]
    n = len(L)
    m_train = float(mse(qa, qb, L, R))
    m_test = float(mse(qa, qb, g["test_left"], g["test_right"]))
    sigma1 = float(np.sqrt(m_train))
    chi2_one = float(chi2(qa, qb, L, R, np.full(n, sigma1)))
    nll_digit = 0.5 * np.log(2 * np.pi * sigma1 ** 2) + np.array([m_train, m_test]) / (
        2 * sigma1 ** 2)
    # one of ten
    s = softmax_run()
    hist = s["history"]
    yt = mnist.load()[3]
    lp = log_softmax(s["test_logits"])
    nt = len(yt)
    right = lp[np.arange(nt), yt]
    loglik10 = float(right.sum())
    ce = float(-right.mean())
    acc10 = float((lp.argmax(1) == yt).mean())
    i = SHOWN_DIGIT
    # one of two
    r = logistic_run()
    lh = r["history"]
    z2, t2 = r["test_logits"], r["test_t"]
    bce = binary_cross_entropy(z2, t2)
    loglik2 = float((t2 * log_sigmoid(z2) + (1 - t2) * log_sigmoid(-z2)).sum())
    acc2 = float(((z2 > 0) == (t2 > 0.5)).mean())
    pair_rows = np.isin(yt, PAIR)
    zt10 = s["test_logits"][pair_rows]
    diff = zt10[:, PAIR[1]] - zt10[:, PAIR[0]]               # the ten-class model on 4 and 9
    t10 = (yt[pair_rows] == PAIR[1]).astype(float)
    bce10 = binary_cross_entropy(diff, t10)
    acc10_pair = float(((diff > 0) == (t10 > 0.5)).mean())
    counts, edges = np.histogram(fits[:, 1], bins=24, range=(b - 4.5 * spread,
                                                              b + 4.5 * spread))
    panels = (
        Panel("Chi-square and the likelihood over one slope", "chi-square and -2 ln L of "
              "the binned right-half ink against the left half's, a straight line, with "
              "the intercept at its best for each slope", "curve",
              dict(x=grid, ys=np.array([c2, -2 * ll]), labels=["chi-square", "-2 ln L"],
                   bins_x=bx, bins_y=by, bins_sigma=bs, bins_n=bn)),
        Panel("The slope in 120 disjoint samples", "the fitted slope in each of 120 "
              "disjoint samples of 500 training digits, against the error read from the "
              "curvature of -ln L", "bars",
              dict(values=counts.astype(float), edges=edges,
                   labels=["%.3f" % v for v in edges[:-1]], errors=fits[:, 3])),
        Panel("One sigma for every digit", "the 500 digits themselves, left-half ink "
              "across, right-half ink up, and the least-squares line", "points",
              dict(xy=np.stack([L, R], 1), line=np.array([qa, qb]))),
        Panel("Cross-entropy while training", "held-out cross-entropy (nats) of the "
              "ten-class softmax regression and of the 4-against-9 logistic regression",
              "curve",
              dict(x=hist[:, 0], ys=np.array([hist[:, 1], hist[:, 2]]),
                   labels=["ten classes, training", "ten classes, held out"],
                   logistic_x=lh[:, 0], logistic_ys=np.array([lh[:, 1], lh[:, 2]]))),
    )
    p_shown = np.exp(lp[i])
    return Study(
        "Every loss is a likelihood", "Chi-square, squared error and cross-entropy, each "
        "minus the log of a likelihood, measured on MNIST", panels,
        dict(source="fits and models made here in numpy on MNIST: a straight line through "
                    "binned ink measurements of 500 seeded training digits, a softmax "
                    "regression on all 60,000 and a logistic regression on the 4s and 9s; "
                    "losses on the 10,000 official test digits",
             dataset="MNIST", evidence=EVIDENCE,
             gauss=dict(n_digits=N_DIGITS, edges=_floats(EDGES), min_bin=MIN_BIN,
                        seed=GAUSS_SEED, points=int(len(bx)), digits_used=int(bn.sum()),
                        a=float(a), b=float(b), sigma_a=float(np.sqrt(cov[0, 0])),
                        sigma_b=sb, chi2_min=float(c2[lo]), dof=int(len(bx) - 2),
                        constant=float(const.mean()), constant_ptp=float(np.ptp(const)),
                        sum_ln_2pi_sigma2=float(np.log(2 * np.pi * bs ** 2).sum()),
                        b_chi2_min=float(grid[lo]), b_likelihood_max=float(grid[hi]),
                        argmin=lo, argmax=hi, loglik_max=float(ll[hi]),
                        curvature=curv, curvature_sigma=float(1 / np.sqrt(curv)),
                        samples=SAMPLES, spread=spread,
                        spread_error=spread / np.sqrt(2 * (SAMPLES - 1)),
                        mean_sigma_b=float(fits[:, 3].mean()),
                        median_chi2_per_dof=float(np.median(fits[:, 4] / (fits[:, 5] - 2))),
                        mean_b=float(fits[:, 1].mean())),
             one_sigma=dict(n=n, a=float(qa), b=float(qb), mse_train=m_train,
                            mse_test=m_test, sigma=sigma1, chi2=chi2_one,
                            n_test=int(len(g["test_left"])),
                            nll_train=float(nll_digit[0]), nll_test=float(nll_digit[1])),
             softmax=dict(sizes=[784, CLASSES], rate=RATE, batch=BATCH, epochs=EPOCHS,
                          seed=SOFTMAX_SEED, steps=int(hist[-1, 0]), parameters=785 * 10,
                          n_train=60000, n_test=nt, cross_entropy=ce,
                          train_cross_entropy=float(hist[-1, 1]), accuracy=acc10,
                          log_likelihood=loglik10,
                          log10_product=loglik10 / np.log(10), uniform=float(np.log(CLASSES)),
                          shown_digit=i, shown_label=int(yt[i]),
                          shown_probabilities=_floats(p_shown),
                          shown_loss=float(-lp[i, yt[i]])),
             logistic=dict(pair=list(PAIR), rate=RATE, batch=BATCH, epochs=EPOCHS,
                           seed=LOGISTIC_SEED, n_train=int(len(pair_data()[1])),
                           n_test=int(len(t2)), nines=int(t2.sum()), bce=bce,
                           train_bce=float(lh[-1, 1]), accuracy=acc2,
                           log_likelihood=loglik2, uniform=float(np.log(2)),
                           ten_class_bce=bce10, ten_class_accuracy=acc10_pair),
             limitations=["a straight line is not quite the shape of the binned ink: the "
                          "shown sample's chi-square is %.1f for %d degrees of freedom"
                          % (float(c2[lo]), len(bx) - 2),
                          "a bin's error is its own sample standard deviation over sqrt n",
                          "one-layer models: the networks later on the page do better",
                          "one seed for each model"]))


STUDIES = {"every-loss-is-a-likelihood": every_loss_is_a_likelihood}

"""Gaussian-process regression in one dimension, in plain numpy.

A GP is a prior over functions: any finite set of function values is jointly
Gaussian with mean zero and a covariance given by a **kernel** k(x, x').  The
kernels here (:class:`RBF`, :class:`Matern32`, :class:`Periodic`,
:class:`Linear`, and their :class:`Sum`) each know their own hyperparameters
and the derivative of the kernel matrix with respect to each of them.

Conditioning on observations ``y = f(X) + noise`` is exact (:func:`posterior`):

    mu*   = k*^T (K + noise I)^-1 y
    cov*  = k** - k*^T (K + noise I)^-1 k*

where ``K = k(X, X)``, ``k* = k(X, x*)`` and ``k** = k(x*, x*)``.  Everything
goes through one Cholesky factor of ``K + noise I`` (:func:`chol`).

The hyperparameters are learned by maximising the **log marginal likelihood**
(:func:`log_marginal`)

    log p(y) = -1/2 y^T K^-1 y  -  1/2 log |K|  -  n/2 log 2 pi
               (data fit)         (complexity)

whose gradient is ``1/2 tr((a a^T - K^-1) dK/dtheta)`` with ``a = K^-1 y``;
:func:`ascend` walks up it in log-parameter space (every hyperparameter is
positive).  :func:`expected_improvement` is the acquisition function the
Bayesian-optimisation animation maximises.  All sampling takes standard
normal draws from outside (:func:`sample`), so the same draws give paths
that move continuously while the hyperparameters change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JITTER = 1e-8           # added to every diagonal before a Cholesky factorisation
LOG2PI = float(np.log(2 * np.pi))


def _col(x):
    """``x`` as a float column ``(n, 1)``: inputs are scalars in one dimension."""
    return np.asarray(x, np.float64).reshape(-1, 1)


# ------------------------------------------------------------------ kernels


class Kernel:
    """Base class: a covariance function with positive hyperparameters in ``self.p``.

    Subclasses implement :meth:`K` and :meth:`dK`; a kernel is called like a
    function, ``k(a, b)``, and ``k1 + k2`` is their :class:`Sum`.
    """

    NAME = ""
    NAMES = ()              # hyperparameter names, in the order they are reported

    def __init__(self, **p):
        unknown = set(p) - set(self.NAMES)
        if unknown:
            raise TypeError("%s has no hyperparameter %s" % (self.NAME, sorted(unknown)))
        self.p = {k: float(p.get(k, 1.0)) for k in self.NAMES}

    def __call__(self, a, b):
        return self.K(a, b)

    def __add__(self, other):
        return Sum(self, other)

    def with_params(self, **changes):
        """A copy with some hyperparameters changed (``k.with_params(lengthscale=2)``)."""
        return type(self)(**dict(self.p, **changes))

    @property
    def params(self):
        """``{name: value}`` of every hyperparameter (a sum prefixes its parts)."""
        return dict(self.p)

    def set_params(self, values):
        """A copy with the hyperparameters in ``values`` (keys as :attr:`params`)."""
        return self.with_params(**{k: v for k, v in values.items() if k in self.NAMES})

    def K(self, a, b):
        """The ``(len(a), len(b))`` matrix of ``k(a_i, b_j)``."""
        raise NotImplementedError

    def dK(self, a, b):
        """``{name: dK/dname}``, each the shape of :meth:`K`."""
        raise NotImplementedError


class RBF(Kernel):
    """Squared exponential: ``v exp(-(a - b)^2 / 2 l^2)``, infinitely smooth."""

    NAME = "RBF"
    NAMES = ("variance", "lengthscale")

    def K(self, a, b):
        r2 = (_col(a) - _col(b).T) ** 2
        return self.p["variance"] * np.exp(-0.5 * r2 / self.p["lengthscale"] ** 2)

    def dK(self, a, b):
        r2 = (_col(a) - _col(b).T) ** 2
        k = self.K(a, b)
        return {"variance": k / self.p["variance"],
                "lengthscale": k * r2 / self.p["lengthscale"] ** 3}


class Matern32(Kernel):
    """Matérn 3/2: ``v (1 + s) exp(-s)`` with ``s = sqrt(3) |a - b| / l``: once
    differentiable, so its samples are rougher than the RBF's."""

    NAME = "Matérn 3/2"
    NAMES = ("variance", "lengthscale")

    def _s(self, a, b):
        return np.sqrt(3.0) * np.abs(_col(a) - _col(b).T) / self.p["lengthscale"]

    def K(self, a, b):
        s = self._s(a, b)
        return self.p["variance"] * (1.0 + s) * np.exp(-s)

    def dK(self, a, b):
        s = self._s(a, b)
        e = np.exp(-s)
        return {"variance": (1.0 + s) * e,
                "lengthscale": self.p["variance"] * s * s * e / self.p["lengthscale"]}


class Periodic(Kernel):
    """``v exp(-2 sin^2(pi (a - b) / p) / l^2)``: repeats exactly every ``period``."""

    NAME = "periodic"
    NAMES = ("variance", "lengthscale", "period")

    def K(self, a, b):
        s2 = np.sin(np.pi * (_col(a) - _col(b).T) / self.p["period"]) ** 2
        return self.p["variance"] * np.exp(-2.0 * s2 / self.p["lengthscale"] ** 2)

    def dK(self, a, b):
        r = _col(a) - _col(b).T
        l, p = self.p["lengthscale"], self.p["period"]
        k = self.K(a, b)
        s2 = np.sin(np.pi * r / p) ** 2
        return {"variance": k / self.p["variance"],
                "lengthscale": k * 4.0 * s2 / l ** 3,
                "period": k * (2.0 * np.pi * r / (l * l * p * p)) * np.sin(2.0 * np.pi * r / p)}


class Linear(Kernel):
    """``v (a - c)(b - c)``: straight lines through ``x = offset``, slopes of
    variance ``v``.  Negative where the two points lie on opposite sides."""

    NAME = "linear"
    NAMES = ("variance", "offset")

    def K(self, a, b):
        c = self.p["offset"]
        return self.p["variance"] * (_col(a) - c) * (_col(b).T - c)

    def dK(self, a, b):
        c = self.p["offset"]
        return {"variance": (_col(a) - c) * (_col(b).T - c),
                "offset": -self.p["variance"] * (_col(a) + _col(b).T - 2.0 * c)}


class Sum(Kernel):
    """``k1 + k2``; hyperparameters are reported as ``"1.name"`` and ``"2.name"``."""

    NAMES = ()

    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2
        self.p = {}

    @property
    def NAME(self):  # mirrors the class attribute of the other kernels
        return "%s + %s" % (self.k1.NAME, self.k2.NAME)

    @property
    def params(self):
        return {**{"1." + k: v for k, v in self.k1.params.items()},
                **{"2." + k: v for k, v in self.k2.params.items()}}

    def set_params(self, values):
        return Sum(self.k1.set_params({k[2:]: v for k, v in values.items() if k[:2] == "1."}),
                   self.k2.set_params({k[2:]: v for k, v in values.items() if k[:2] == "2."}))

    def with_params(self, **changes):
        return self.set_params(changes)

    def K(self, a, b):
        return self.k1.K(a, b) + self.k2.K(a, b)

    def dK(self, a, b):
        return {**{"1." + k: v for k, v in self.k1.dK(a, b).items()},
                **{"2." + k: v for k, v in self.k2.dK(a, b).items()}}


# ------------------------------------------------------------------ linear algebra


def chol(K, jitter=JITTER):
    """Lower Cholesky factor of ``K + jitter I`` (the jitter keeps a nearly
    singular kernel matrix factorisable)."""
    K = np.asarray(K, np.float64)
    return np.linalg.cholesky(K + jitter * np.eye(len(K)))


def chol_solve(L, b):
    """``K^-1 b`` from the Cholesky factor ``L`` of ``K`` (two triangular solves)."""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def gram(kernel, X, noise):
    """``K + diag(noise)``: the covariance of the observations.  ``noise`` is a
    variance, one number or one per observation."""
    X = np.asarray(X, np.float64)
    return kernel(X, X) + np.diag(np.broadcast_to(np.asarray(noise, np.float64), X.shape))


def sample(mu, cov, z):
    """Function values ``mu + L z`` for standard normal draws ``z`` ``(..., n)``:
    one sample path per row, exact draws from ``N(mu, cov)``.

    ``cov`` gets :data:`JITTER` on its diagonal so a prior over a dense grid
    (which is numerically singular) can be factorised.
    """
    L = chol(cov)
    return np.asarray(mu) + np.asarray(z) @ L.T


# ------------------------------------------------------------------ prior and posterior


@dataclass
class Posterior:
    """The GP conditioned on data, evaluated on the grid ``xs``.

    ``mu`` and ``sd`` are the posterior mean and standard deviation on the
    grid, ``cov`` its covariance; ``K`` (noise included), ``ks`` ``(n, m)``
    and ``kss`` ``(m, m)`` are the three kernel matrices of the formula,
    ``alpha = K^-1 y`` and ``L`` the Cholesky factor of ``K``.
    """

    xs: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    K: np.ndarray
    ks: np.ndarray
    kss: np.ndarray
    alpha: np.ndarray
    L: np.ndarray

    @property
    def sd(self):
        """Posterior standard deviation on the grid (never below 0)."""
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def samples(self, z):
        """Posterior sample paths for standard normal draws ``z`` ``(..., m)``."""
        return sample(self.mu, self.cov, z)


def prior(kernel, xs):
    """The prior on the grid ``xs``: :class:`Posterior` with no data (``mu`` = 0,
    ``cov`` = ``k(xs, xs)``; its data matrices are empty)."""
    xs = np.asarray(xs, np.float64)
    kss = kernel(xs, xs)
    empty = np.zeros((0, 0))
    return Posterior(xs, np.zeros(len(xs)), kss, empty, np.zeros((0, len(xs))), kss,
                     np.zeros(0), empty)


def posterior(kernel, X, y, xs, noise):
    """Condition the GP on ``y`` at ``X`` (observation variance ``noise``: one
    number, or one per point) and evaluate on ``xs``.  See :class:`Posterior`."""
    X, y, xs = (np.asarray(v, np.float64) for v in (X, y, xs))
    if len(X) == 0:
        return prior(kernel, xs)
    K = gram(kernel, X, noise)
    ks, kss = kernel(X, xs), kernel(xs, xs)
    L = chol(K)
    alpha = chol_solve(L, y)
    v = np.linalg.solve(L, ks)                  # L^-1 k*, so k*^T K^-1 k* = v^T v
    return Posterior(xs, ks.T @ alpha, kss - v.T @ v, K, ks, kss, alpha, L)


# ------------------------------------------------------------------ marginal likelihood


@dataclass
class Evidence:
    """``log p(y)`` and its parts: ``fit = -1/2 y^T K^-1 y``, ``complexity =
    -1/2 log |K|``, ``const = -n/2 log 2 pi``; ``grad`` is d log p(y) / d
    hyperparameter for every kernel hyperparameter and for ``"noise"``."""

    fit: float
    complexity: float
    const: float
    grad: dict

    @property
    def value(self):
        """``log p(y)``."""
        return self.fit + self.complexity + self.const


def log_marginal(kernel, X, y, noise):
    """The log marginal likelihood of ``y`` at ``X`` under ``kernel`` with
    observation variance ``noise`` (a number), and its gradient."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    K = gram(kernel, X, noise)
    L = chol(K)
    alpha = chol_solve(L, y)
    Kinv = chol_solve(L, np.eye(len(X)))
    fit = -0.5 * float(y @ alpha)
    complexity = -float(np.log(np.diag(L)).sum())          # -1/2 log|K| = -sum log L_ii
    W = np.outer(alpha, alpha) - Kinv                      # d log p / dK = 1/2 W
    grad = {k: 0.5 * float(np.sum(W * dK)) for k, dK in kernel.dK(X, X).items()}
    grad["noise"] = 0.5 * float(np.trace(W))
    return Evidence(fit, complexity, -0.5 * len(X) * LOG2PI, grad)


def ascend(kernel, X, y, noise, names, lr, steps, noise_name="noise"):
    """Gradient ascent on ``log p(y)`` in log-parameter space.

    ``names`` lists the hyperparameters that move (kernel names and/or
    ``"noise"``); the others stay.  Returns ``(kernels, noises, evidences)``:
    the kernel, the noise variance and the :class:`Evidence` before every step
    and after the last (``steps + 1`` of each).  Ascent on ``log theta`` keeps
    every hyperparameter positive and makes one learning rate serve
    quantities of different size: ``d log p / d log theta = theta d log p / d
    theta``.
    """
    kernels, noises, evs = [kernel], [float(noise)], []
    for _ in range(steps):
        ev = log_marginal(kernel, X, y, noise)
        evs.append(ev)
        p = dict(kernel.params, **{noise_name: noise})
        for k in names:
            p[k] = p[k] * np.exp(lr * p[k] * ev.grad[k])
        noise = p.pop(noise_name)
        kernel = kernel.set_params(p)
        kernels.append(kernel)
        noises.append(float(noise))
    evs.append(log_marginal(kernel, X, y, noise))
    return kernels, noises, evs


# ------------------------------------------------------------------ Bayesian optimisation


def norm_pdf(z):
    """Standard normal density."""
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def norm_cdf(z):
    """Standard normal distribution function (via the complementary error
    function, which numpy lacks: a rational approximation good to 1e-7)."""
    # Abramowitz & Stegun 7.1.26
    z = np.asarray(z, np.float64)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(z) / np.sqrt(2))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (
        -1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z / 2)
    return 0.5 * (1.0 + np.sign(z) * erf)


def expected_improvement(mu, sd, best, xi=0.0):
    """``E[max(f - best - xi, 0)]`` for ``f ~ N(mu, sd^2)``, for maximisation:

        EI = (mu - best - xi) Phi(z) + sd phi(z),  z = (mu - best - xi) / sd

    (0 where ``sd`` is 0).  Large where the mean is high (exploit) and where
    the uncertainty is large (explore).
    """
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    d = mu - best - xi
    safe = np.where(sd > 1e-12, sd, 1.0)
    z = d / safe
    ei = d * norm_cdf(z) + safe * norm_pdf(z)
    return np.where(sd > 1e-12, ei, 0.0)

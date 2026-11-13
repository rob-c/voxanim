"""What the generalisation animations measure, in numpy, on the deep MLP of
:mod:`voxanim.deepnet`.

* **Label noise** -- :func:`corrupt` replaces a seeded fraction of the labels
  with a random *wrong* class (so the fraction stated is exactly the fraction
  of labels that are false).
* **Errors** -- :func:`predictions`, :func:`error_rate`, :func:`confusion`
  score a trained network on a whole set (chunked forward passes), and
  :func:`misclassified` lists the examples it gets wrong.
* **Augmentation** -- :class:`Augment` is a random affine transform of a
  batch of 28 x 28 images: a shift of up to ``shift`` pixels in each
  direction, a rotation of up to ``degrees`` and a scale in ``1 +- scale``,
  resampled with bilinear interpolation (:func:`affine`).  It is the
  ``augment`` hook of :func:`voxanim.deepnet.train`.
* **Smoothness** -- :func:`weight_norm` and :func:`sensitivity` (how much the
  output distribution moves under input noise of a given std) measure how
  smooth the function a network computes is.
* **Slices of the loss** -- :func:`along_line` evaluates the loss on the
  straight line between two parameter sets, :func:`around` along random unit
  directions about one, :func:`perturbed` after noise of a given norm is
  added to the weights, and :func:`plane_grid` on a plane through two minima.
"""

from __future__ import annotations

import numpy as np

from .deepnet import cross_entropy, forward

CHUNK = 1000                 # rows per forward pass when scoring a set
SIDE = 28                    # MNIST images are SIDE x SIDE


# ------------------------------------------------------------------ labels and scores


def corrupt(y, fraction, classes, rng):
    """``(labels, mask)``: ``y`` with a ``fraction`` of its entries (chosen by
    ``rng``) replaced by a random *other* class; ``mask`` is True where a
    label was changed."""
    y = np.asarray(y).copy()
    n = len(y)
    k = int(round(fraction * n))
    idx = rng.permutation(n)[:k]
    # a random offset in 1..classes-1 never lands on the true label
    y[idx] = (y[idx] + rng.integers(1, classes, k)) % classes
    mask = np.zeros(n, bool)
    mask[idx] = True
    return y, mask


def probabilities(spec, P, x, stats=None):
    """``(N, classes)`` softmax outputs on ``x``, in chunks."""
    return np.concatenate([forward(spec, P, x[s:s + CHUNK], stats=stats)["p"]
                           for s in range(0, len(x), CHUNK)])


def predictions(spec, P, x, stats=None):
    """``(N,)`` predicted classes on ``x``."""
    return probabilities(spec, P, x, stats).argmax(1)


def error_rate(spec, P, x, y, stats=None):
    """Fraction of ``x`` classified wrongly."""
    return float((predictions(spec, P, x, stats) != np.asarray(y)).mean())


def loss_on(spec, P, x, y, classes, stats=None):
    """Mean cross-entropy on ``x`` against integer labels ``y``."""
    return cross_entropy(probabilities(spec, P, x, stats), np.eye(classes)[np.asarray(y)])


def confusion(pred, y, classes):
    """``(classes, classes)`` counts: row the true class, column the prediction."""
    m = np.zeros((classes, classes), int)
    np.add.at(m, (np.asarray(y), np.asarray(pred)), 1)
    return m


def misclassified(pred, y):
    """Indices of the examples whose prediction is not their label."""
    return np.nonzero(np.asarray(pred) != np.asarray(y))[0]


# ------------------------------------------------------------------ augmentation


def affine(x, dx, dy, angle, scale):
    """Images ``x`` ``(N, 784)`` each moved by ``(dx, dy)`` pixels, rotated by
    ``angle`` radians about the centre and magnified by ``scale`` (arrays of
    ``N``), resampled bilinearly; pixels pulled in from outside are 0.

    The output pixel at ``(r, c)`` takes its value from the input at the
    inverse-mapped point, so a positive ``dx`` moves the digit to the right.
    """
    x = np.asarray(x, np.float64).reshape(-1, SIDE, SIDE)
    n = len(x)
    dx, dy, angle, scale = (np.broadcast_to(np.asarray(v, np.float64), (n,))
                            for v in (dx, dy, angle, scale))
    c = (SIDE - 1) / 2.0
    rr, cc = np.meshgrid(np.arange(SIDE) - c, np.arange(SIDE) - c, indexing="ij")
    cos, sin = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
    k = 1.0 / scale[:, None, None]
    # inverse map: undo the shift, then the rotation and the scale
    u = cc[None] - dx[:, None, None]
    v = rr[None] - dy[:, None, None]
    src_c = k * (cos * u + sin * v) + c
    src_r = k * (-sin * u + cos * v) + c
    r0 = np.floor(src_r).astype(int)
    c0 = np.floor(src_c).astype(int)
    fr, fc = src_r - r0, src_c - c0
    out = np.zeros((n, SIDE, SIDE))
    rows = np.arange(n)[:, None, None]
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                      (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        r, cq = r0 + dr, c0 + dc
        inside = (r >= 0) & (r < SIDE) & (cq >= 0) & (cq < SIDE)
        out += w * np.where(inside, x[rows, np.clip(r, 0, SIDE - 1), np.clip(cq, 0, SIDE - 1)],
                            0.0)
    return out.reshape(n, SIDE * SIDE)


class Augment:
    """Random affine transforms of a batch: shift uniform in ``+-shift`` px
    (each axis), rotation uniform in ``+-degrees``, scale uniform in
    ``1 +- scale``.  ``draw`` gives the parameters, ``__call__`` applies
    them; ``settings`` keys a cache file."""

    def __init__(self, shift=3.0, degrees=15.0, scale=0.15):
        self.shift, self.degrees, self.scale = float(shift), float(degrees), float(scale)

    def settings(self):
        """The JSON-able dict that keys a cache file."""
        return dict(kind="affine", shift=self.shift, degrees=self.degrees, scale=self.scale)

    def draw(self, rng, n):
        """``(dx, dy, angle, scale)`` for ``n`` images, from ``rng``."""
        dx = rng.uniform(-self.shift, self.shift, n)
        dy = rng.uniform(-self.shift, self.shift, n)
        angle = np.radians(rng.uniform(-self.degrees, self.degrees, n))
        scale = rng.uniform(1 - self.scale, 1 + self.scale, n)
        return dx, dy, angle, scale

    def __call__(self, rng, x):
        """The batch ``x`` ``(N, 784)``, each image transformed at random."""
        return affine(x, *self.draw(rng, len(x)))


# ------------------------------------------------------------------ smoothness


def weight_norm(P):
    """The Euclidean norm of all the weight matrices together."""
    return float(np.sqrt(sum((v * v).sum() for k, v in P.items() if k.startswith("W"))))


def sensitivity(spec, P, x, rng, sigma, stats=None):
    """Mean over ``x`` of ``|p(x + e) - p(x)|_1`` for pixel noise ``e ~ N(0,
    sigma^2)``: how far the output distribution moves when the input is
    jiggled -- small for a smooth function."""
    p0 = probabilities(spec, P, x, stats)
    p1 = probabilities(spec, P, x + rng.normal(0.0, sigma, x.shape), stats)
    return float(np.abs(p1 - p0).sum(1).mean())


# ------------------------------------------------------------------ slices of the loss


def interpolate(Pa, Pb, u):
    """``(1 - u) Pa + u Pb``, key by key."""
    return {k: (1 - u) * Pa[k] + u * Pb[k] for k in Pa}


def param_distance(Pa, Pb):
    """Euclidean distance between two parameter sets."""
    return float(np.sqrt(sum(((Pa[k] - Pb[k]) ** 2).sum() for k in Pa)))


def random_direction(P, rng):
    """A unit vector in parameter space (a dict like ``P``), Gaussian in every entry."""
    d = {k: rng.normal(0.0, 1.0, v.shape) for k, v in P.items()}
    n = np.sqrt(sum((v * v).sum() for v in d.values()))
    return {k: v / n for k, v in d.items()}


def moved(P, d, r):
    """``P + r d``."""
    return {k: P[k] + r * d[k] for k in P}


def along_line(spec, Pa, Pb, us, x, y, classes):
    """The loss on ``x, y`` at ``(1 - u) Pa + u Pb`` for each ``u`` in ``us``."""
    return np.array([loss_on(spec, interpolate(Pa, Pb, u), x, y, classes) for u in us])


def around(spec, P, rng, radii, x, y, classes, directions=4):
    """``(directions, len(radii))`` the loss on ``x, y`` at ``P + r d`` for
    random unit directions ``d`` and each radius ``r`` (negative radii allowed)."""
    out = []
    for _ in range(directions):
        d = random_direction(P, rng)
        out.append([loss_on(spec, moved(P, d, r), x, y, classes) for r in radii])
    return np.array(out)


def perturbed(spec, P, rng, norm, x, y, trials=5):
    """Mean error rate on ``x, y`` after Gaussian noise of Euclidean norm
    ``norm`` is added to all the parameters, over ``trials`` draws."""
    return float(np.mean([error_rate(spec, moved(P, random_direction(P, rng), norm), x, y)
                          for _ in range(trials)]))


def plane_grid(spec, Pa, Pb, d, us, vs, x, y, classes):
    """``(len(us), len(vs))`` the loss on ``x, y`` over the plane
    ``Pa + u (Pb - Pa) + v d``: the line between two minima along ``u`` and a
    direction ``d`` (a parameter dict) across it."""
    diff = {k: Pb[k] - Pa[k] for k in Pa}
    L = np.zeros((len(us), len(vs)))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            Q = {k: Pa[k] + u * diff[k] + v * d[k] for k in Pa}
            L[i, j] = loss_on(spec, Q, x, y, classes)
    return L

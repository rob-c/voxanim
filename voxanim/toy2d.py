"""Seeded two-dimensional datasets and small, hand-differentiated NumPy models.

The studies deliberately report their measured outcomes, including imperfect fits.
Nothing in this module relies on scikit-learn or on manufactured training curves.
"""

from __future__ import annotations

import itertools

import numpy as np

from .optim import Adam


def dataset(kind="moons", n=512, seed=51, noise=0.08):
    """Return points and ground-truth component labels."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % (8 if kind == "gaussians" else 2)
    rng.shuffle(y)
    if kind == "gaussians":
        a = y * np.pi / 4
        x = 2 * np.column_stack((np.cos(a), np.sin(a)))
    elif kind == "rings":
        a = rng.uniform(0, 2 * np.pi, n)
        r = 0.8 + y * 1.0
        x = np.column_stack((r * np.cos(a), r * np.sin(a)))
    elif kind == "spirals":
        a = rng.uniform(0.2, 2.8 * np.pi, n)
        r = a / (2.8 * np.pi) * 2
        a = a + y * np.pi
        x = np.column_stack((r * np.cos(a), r * np.sin(a)))
    elif kind == "moons":
        a = rng.uniform(0, np.pi, n)
        x = np.column_stack((np.cos(a), np.sin(a)))
        x[y == 1] = np.column_stack((1 - np.cos(a[y == 1]), 0.5 - np.sin(a[y == 1])))
        x -= [0.5, 0.25]
    else:
        raise ValueError(kind)
    return x + rng.normal(0, noise, x.shape), y


def softmax(z):
    p = np.exp(z - z.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def init(sizes, rng):
    """Tanh MLP, linear output; fan-in scaled initialization."""
    out = {}
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        out[f"W{i}"] = rng.normal(0, 1 / np.sqrt(a), (a, b))
        out[f"b{i}"] = np.zeros(b)
    return out


def forward(p, x):
    a = [x]
    n = len(p) // 2
    for i in range(n):
        z = a[-1] @ p[f"W{i}"] + p[f"b{i}"]
        a.append(np.tanh(z) if i < n - 1 else z)
    return a[-1], a


def backward(p, a, d):
    g = {}
    for i in reversed(range(len(p) // 2)):
        if i < len(p) // 2 - 1:
            d = d * (1 - a[i + 1] ** 2)
        g[f"W{i}"], g[f"b{i}"] = a[i].T @ d, d.sum(0)
        d = d @ p[f"W{i}"].T
    return g, d


def loss_grads(p, x, y, classification=False):
    z, a = forward(p, x)
    if classification:
        pr = softmax(z)
        loss = -np.log(np.maximum(pr[np.arange(len(y)), y], 1e-300)).mean()
        d = pr.copy()
        d[np.arange(len(y)), y] -= 1
        d /= len(y)
    else:
        loss = np.mean((z - y) ** 2)
        d = 2 * (z - y) / z.size
    g, _ = backward(p, a, d)
    return float(loss), g


def fit(x, y, widths=(32, 32), steps=500, lr=0.005, seed=51, classification=False):
    rng = np.random.default_rng(seed)
    k = int(y.max()) + 1 if classification else y.shape[1]
    p = init((x.shape[1], *widths, k), rng)
    opt, losses = Adam(p, lr), []
    for _ in range(steps):
        idx = rng.choice(len(x), min(96, len(x)), replace=False)
        loss, g = loss_grads(p, x[idx], y[idx], classification)
        opt.step(p, g)
        losses.append(loss)
    return p, np.array(losses)


def kernel(x, z, gamma=1.0):
    return np.exp(-gamma * ((x[:, None] - z[None]) ** 2).sum(-1))


def svm(x, labels, c=2.0, gamma=1.0, passes=12, seed=51):
    """Binary RBF SVM, sequential minimal optimisation of the dual."""
    y = 2.0 * labels - 1
    a, b = np.zeros(len(y)), 0.0
    k = kernel(x, x, gamma)
    rng = np.random.default_rng(seed)
    for _ in range(passes):
        changed = 0
        for i in rng.permutation(len(x)):
            e = k @ (a * y) + b - y
            if not ((y[i] * e[i] < -1e-3 and a[i] < c) or (y[i] * e[i] > 1e-3 and a[i] > 0)):
                continue
            candidates = np.argsort(-np.abs(e[i] - e))
            for j in candidates:
                if i == j:
                    continue
                ai, aj = a[i], a[j]
                if y[i] != y[j]:
                    lo, hi = max(0, aj - ai), min(c, c + aj - ai)
                else:
                    lo, hi = max(0, ai + aj - c), min(c, ai + aj)
                eta = 2 * k[i, j] - k[i, i] - k[j, j]
                if hi <= lo or eta >= -1e-12:
                    continue
                newj = np.clip(aj - y[j] * (e[i] - e[j]) / eta, lo, hi)
                if abs(newj - aj) < 1e-8:
                    continue
                a[j] = newj
                a[i] = ai + y[i] * y[j] * (aj - newj)
                b1 = b - e[i] - y[i] * (a[i] - ai) * k[i, i] - y[j] * (a[j] - aj) * k[i, j]
                b2 = b - e[j] - y[i] * (a[i] - ai) * k[i, j] - y[j] * (a[j] - aj) * k[j, j]
                b = b1 if 0 < a[i] < c else b2 if 0 < a[j] < c else (b1 + b2) / 2
                changed += 1
                break
        if changed == 0:
            break
    return dict(x=x, y=y, a=a, b=b, gamma=gamma)


def svm_predict(model, x):
    return kernel(x, model["x"], model["gamma"]) @ (model["a"] * model["y"]) + model["b"]


def flow_init(rng, layers=6, hidden=24):
    p = {}
    for l in range(layers):
        p[f"W{l}"] = rng.normal(0, 0.2, (1, hidden))
        p[f"b{l}"] = np.zeros(hidden)
        p[f"V{l}"] = rng.normal(0, 0.01, (hidden, 2))
        p[f"c{l}"] = np.zeros(2)
    return p


def flow_forward(p, x):
    z, logdet, tape = x.copy(), np.zeros(len(x)), []
    for l in range(len(p) // 4):
        i, j = l % 2, 1 - l % 2
        h = np.tanh(z[:, i : i + 1] @ p[f"W{l}"] + p[f"b{l}"])
        st = h @ p[f"V{l}"] + p[f"c{l}"]
        s = 0.7 * np.tanh(st[:, 0])
        out = z.copy()
        out[:, j] = z[:, j] * np.exp(s) + st[:, 1]
        tape.append((z, h, st, s))
        z = out
        logdet += s
    return z, logdet, tape


def flow_loss_grads(p, x):
    z, ld, tape = flow_forward(p, x)
    loss = np.mean(0.5 * (z**2).sum(1) + np.log(2 * np.pi) - ld)
    dz, g = z / len(z), {}
    for l in reversed(range(len(tape))):
        i, j = l % 2, 1 - l % 2
        old, h, st, s = tape[l]
        ds = dz[:, j] * old[:, j] * np.exp(s) - 1 / len(z)
        dst = np.column_stack((ds * 0.7 * (1 - np.tanh(st[:, 0]) ** 2), dz[:, j]))
        g[f"V{l}"], g[f"c{l}"] = h.T @ dst, dst.sum(0)
        dh = (dst @ p[f"V{l}"].T) * (1 - h**2)
        g[f"W{l}"], g[f"b{l}"] = old[:, i : i + 1].T @ dh, dh.sum(0)
        d = dz.copy()
        d[:, j] *= np.exp(s)
        d[:, i] += (dh @ p[f"W{l}"].T)[:, 0]
        dz = d
    return float(loss), g


def flow_inverse(p, z, until=0):
    x = z.copy()
    for l in reversed(range(until, len(p) // 4)):
        i, j = l % 2, 1 - l % 2
        h = np.tanh(x[:, i : i + 1] @ p[f"W{l}"] + p[f"b{l}"])
        st = h @ p[f"V{l}"] + p[f"c{l}"]
        s = 0.7 * np.tanh(st[:, 0])
        x[:, j] = (x[:, j] - st[:, 1]) * np.exp(-s)
    return x


def gmm_em(x, k=3, steps=80, seed=51):
    rng = np.random.default_rng(seed)
    mu = x[rng.choice(len(x), k, replace=False)].copy()
    cov = np.repeat(np.eye(x.shape[1])[None], k, axis=0)
    pi = np.ones(k) / k
    history, means = [], []
    for _ in range(steps):
        lp = []
        for j in range(k):
            d = x - mu[j]
            lp.append(
                np.log(pi[j])
                - 0.5
                * (
                    np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov[j]), d)
                    + np.linalg.slogdet(cov[j])[1]
                    + x.shape[1] * np.log(2 * np.pi)
                )
            )
        lp = np.array(lp).T
        mx = lp.max(1, keepdims=True)
        r = np.exp(lp - mx)
        total = r.sum(1, keepdims=True)
        r /= total
        history.append(float(np.mean(mx + np.log(total))))
        means.append(mu.copy())
        nk = r.sum(0)
        pi = nk / len(x)
        mu = r.T @ x / nk[:, None]
        for j in range(k):
            d = x - mu[j]
            cov[j] = (d.T * r[:, j]) @ d / nk[j] + np.eye(x.shape[1]) * 1e-5
    return dict(
        mu=mu, cov=cov, pi=pi, responsibility=r, loss=np.array(history), means=np.array(means)
    )


def tree_fit(x, y, depth=3, min_leaf=8):
    """Least-squares CART tree; threshold search over observed midpoints."""
    if depth == 0 or len(y) < 2 * min_leaf:
        return float(y.mean())
    best = None
    for f in range(x.shape[1]):
        order = np.argsort(x[:, f])
        xs, ys = x[order, f], y[order]
        cs, cs2 = np.cumsum(ys), np.cumsum(ys**2)
        for n in range(min_leaf, len(y) - min_leaf + 1):
            if xs[n - 1] == xs[n]:
                continue
            score = cs2[-1] - cs[n - 1] ** 2 / n - (cs[-1] - cs[n - 1]) ** 2 / (len(y) - n)
            if best is None or score < best[0]:
                best = (score, f, (xs[n - 1] + xs[n]) / 2)
    if best is None:
        return float(y.mean())
    _, f, threshold = best
    mask = x[:, f] <= threshold
    return (
        f,
        threshold,
        tree_fit(x[mask], y[mask], depth - 1, min_leaf),
        tree_fit(x[~mask], y[~mask], depth - 1, min_leaf),
    )


def tree_predict(tree, x):
    if isinstance(tree, float):
        return np.full(len(x), tree)
    f, threshold, left, right = tree
    mask = x[:, f] <= threshold
    out = np.empty(len(x))
    out[mask], out[~mask] = tree_predict(left, x[mask]), tree_predict(right, x[~mask])
    return out


def linear_svm_loss(p, x, y, c=1.0):
    """Primal linear soft-margin objective with labels in {-1, +1}."""
    margins = y * (x @ p["w"] + p["b"])
    active = margins < 1
    loss = 0.5 * np.sum(p["w"] ** 2) + c * np.maximum(0, 1 - margins).mean()
    d = -c * y * active / len(y)
    return float(loss), dict(w=p["w"] + x.T @ d, b=np.array(d.sum()))


def linear_svm(x, y, c=1.0, steps=600):
    p = dict(w=np.zeros(x.shape[1]), b=np.array(0.0))
    opt = Adam(p, 0.01)
    for _ in range(steps):
        _, g = linear_svm_loss(p, x, 2 * y - 1, c)
        opt.step(p, g)
    return p

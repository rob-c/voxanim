"""Measured studies for the unpublished systems, optimisation and RL lessons.

All arrays are produced by seeded experiments or exact arithmetic. Timings are
local observations, never hardware specifications. Synthetic tasks are labelled.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
_RUN_EVIDENCE = []


def _cache(name, build, **settings):
    from . import realdata

    settings = dict(
        version=2,
        seed=17,
        source=realdata.source_hash("next_systems.py", "realdata.py", "optim.py"),
        **settings,
    )
    result = {k: np.asarray(v) for k, v in cached_npz("next-" + name, settings, build).items()}
    trained = {
        "edge-stability",
        "batch-noise",
        "weight-average",
        "infonce-temperature",
        "hard-negatives",
        "neural-cellular",
        "evolution-strategies",
    }
    if name in trained:
        from .mnist import SHA256, URL

        params = {k: v for k, v in result.items() if k.startswith("P_")}
        if name == "neural-cellular":
            params.update({k: result[k] for k in ("A", "B", "b", "c") if k in result})
        info = realdata.save_evidence(
            name,
            params,
            {k: v for k, v in result.items() if k not in params},
            dict(
                dataset="MNIST",
                raw_sha256=SHA256,
                source_url=URL,
                preprocessing="See generating function and saved experiment arrays",
            ),
            settings,
            ("next_systems.py", "optim.py"),
        )
        _RUN_EVIDENCE.append(info)
    return result


def _curve(title, caption, ys, labels, x=None, notes=()):
    ys = np.atleast_2d(ys)
    return Panel(
        title,
        caption,
        "curve",
        dict(x=np.arange(ys.shape[1]) if x is None else x, ys=ys, labels=list(labels)),
        tuple(notes),
    )


def _bars(title, caption, values, labels, notes=()):
    return Panel(
        title, caption, "bars", dict(values=np.asarray(values), labels=list(labels)), tuple(notes)
    )


def _matrix(title, caption, values, notes=()):
    return Panel(title, caption, "matrix", dict(values=np.asarray(values)), tuple(notes))


def _points(title, caption, xy, classes=None, notes=()):
    data = dict(xy=np.asarray(xy))
    if classes is not None:
        data["classes"] = np.asarray(classes)
    return Panel(title, caption, "points", data, tuple(notes))


def _images(title, caption, values, labels, notes=()):
    return Panel(
        title, caption, "images", dict(values=np.asarray(values), labels=list(labels)), tuple(notes)
    )


def _study(title, subtitle, panels, **meta):
    return Study(title, subtitle, tuple(panels), dict(seed=17, **meta))


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mlp_init(d, h, c, rng):
    return dict(
        W=rng.normal(0, np.sqrt(2 / d), (d, h)),
        b=np.zeros(h),
        U=rng.normal(0, 1 / np.sqrt(h), (h, c)),
        v=np.zeros(c),
    )


def mlp_loss(P, x, y, grads=True):
    """Tanh classifier, with a complete analytic backward pass."""
    h = np.tanh(x @ P["W"] + P["b"])
    z = h @ P["U"] + P["v"]
    p = softmax(z)
    loss = float(np.mean(np.logaddexp.reduce(z, axis=1) - z[np.arange(len(y)), y]))
    if not grads:
        return loss, float(np.mean(p.argmax(1) == y))
    dz = p.copy()
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    dh = (dz @ P["U"].T) * (1 - h * h)
    return loss, dict(W=x.T @ dh, b=dh.sum(0), U=h.T @ dz, v=dz.sum(0))


def _digit_data(n=320, test=256, side=7):
    from .mnist import load

    x, y, xt, yt = load()

    def pool(a):
        return a.reshape(-1, side, 28 // side, side, 28 // side).mean((2, 4)).reshape(-1, side**2)

    return pool(x[:n]), y[:n].astype(int), pool(xt[:test]), yt[:test].astype(int)


def _pack(P):
    return np.concatenate([v.ravel() for v in P.values()])


def _unpack(v, P):
    out, at = {}, 0
    for k, p in P.items():
        out[k] = v[at : at + p.size].reshape(p.shape)
        at += p.size
    return out


def _sharpness(P, x, y, iterations=8):
    """Hessian-vector products from symmetric differences of exact gradients."""
    p = _pack(P)
    v = np.ones(len(p)) / np.sqrt(len(p))

    def hv(q):
        eps = 1e-4
        gp = _pack(mlp_loss(_unpack(p + eps * q, P), x, y)[1])
        gm = _pack(mlp_loss(_unpack(p - eps * q, P), x, y)[1])
        return (gp - gm) / (2 * eps)

    for _ in range(iterations):
        u = hv(v)
        v = u / max(np.linalg.norm(u), 1e-12)
    return float(v @ hv(v))


def newton():
    def build():
        H = np.array([[50.5, 49.5], [49.5, 50.5]])
        paths, losses = [], []
        for method in ("GD", "Newton"):
            w = np.array([2.0, -1.0])
            path, loss = [], []
            for _ in range(101):
                path.append(w.copy())
                loss.append(w @ H @ w / 2)
                w -= 0.018 * (H @ w) if method == "GD" else np.linalg.solve(H, H @ w)
            paths.append(path)
            losses.append(loss)
        rosen = []
        for method in ("GD", "Newton"):
            w = np.array([-1.2, 1.0])
            ls = []
            for _ in range(201):
                a, b = w
                loss = (1 - a) ** 2 + 100 * (b - a * a) ** 2
                ls.append(loss)
                g = np.array([2 * (a - 1) - 400 * a * (b - a * a), 200 * (b - a * a)])
                h = np.array([[2 - 400 * b + 1200 * a * a, -400 * a], [-400 * a, 200.0]])
                step = 0.001 * g if method == "GD" else np.linalg.solve(h, g)
                scale = 1.0
                while scale > 1e-9:
                    z = w - scale * step
                    if (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2 <= loss:
                        break
                    scale /= 2
                w -= scale * step
            rosen.append(ls)
        return dict(H=H, paths=paths, losses=losses, rosen=rosen)

    d = _cache("newton", build, steps=201)
    return _study(
        "Newton versus gradient descent",
        "Exact quadratic and Rosenbrock objectives",
        [
            _matrix(
                "A tilted quadratic",
                "Off-diagonal curvature couples the two coordinates.",
                d["H"],
                ("Hessian condition number: %.0f" % np.linalg.cond(d["H"]),),
            ),
            _points(
                "Paths through the quadratic",
                "Newton solves the quadratic in one step.",
                d["paths"].reshape(-1, 2),
                np.repeat([0, 1], 101),
            ),
            _curve(
                "Quadratic objective",
                "Gradient descent must respect the steepest direction.",
                d["losses"],
                ["GD", "Newton"],
            ),
            _curve(
                "Rosenbrock needs a line search",
                "The same Newton rule needs damping on a curved valley.",
                d["rosen"],
                ["GD", "damped Newton"],
                notes=("Final Newton loss: %.3g" % d["rosen"][1, -1],),
            ),
        ],
    )


def edge_stability():
    def build():
        x, y, _, _ = _digit_data(256)
        traces = []
        sharp = []
        params = {}
        for lr in (1.0, 3.0):
            P = mlp_init(49, 16, 10, np.random.default_rng(17))
            ls = []
            hs = []
            for step in range(201):
                loss, G = mlp_loss(P, x, y)
                ls.append(loss)
                if step % 5 == 0:
                    hs.append(_sharpness(P, x, y))
                if step == 200:
                    break
                for k in P:
                    P[k] -= lr * G[k]
            traces.append(ls)
            sharp.append(hs)
            params.update({"P_%s_%s" % (lr, k): v for k, v in P.items()})
        return dict(loss=traces, sharp=sharp, lrs=[1.0, 3.0], **params)

    d = _cache("edge-stability", build, updates=200, observations=201, n=256, hidden=16)
    ratio = d["sharp"] * d["lrs"][:, None] / 2
    return _study(
        "The edge of stability",
        "MNIST: 7×7 pooled pixels, tanh MLP, full-batch GD",
        [
            _curve(
                "Full-batch training",
                "Large steps can oscillate while the average loss still falls.",
                d["loss"],
                ["η = 1", "η = 3"],
            ),
            _curve(
                "Measured top curvature",
                "Power iteration uses finite-difference Hessian-vector products.",
                d["sharp"],
                ["η = 1", "η = 3"],
                np.arange(0, 201, 5),
            ),
            _curve(
                "Curvature relative to the boundary",
                "The quadratic stability threshold is ηλ / 2 = 1.",
                np.vstack([ratio, np.ones(41)]),
                ["η = 1", "η = 3", "boundary"],
                np.arange(0, 201, 5),
            ),
            _bars(
                "What this run measured",
                "A short run need not settle exactly on the proposed boundary.",
                ratio[:, -1],
                ["η = 1", "η = 3"],
                ("Final ηλ/2: %.3f and %.3f" % tuple(ratio[:, -1]),),
            ),
        ],
    )


def batch_noise():
    def build():
        x, y, xt, yt = _digit_data(512)
        P = mlp_init(49, 20, 10, np.random.default_rng(17))
        rng = np.random.default_rng(18)
        full = _pack(mlp_loss(P, x, y)[1])
        batches = np.array([1, 4, 16, 64, 256])
        noise = []
        losses = []
        params = {}
        for b in batches:
            noise.append(
                np.mean(
                    [
                        np.linalg.norm(_pack(mlp_loss(P, x[r], y[r])[1]) - full) ** 2
                        for r in rng.integers(0, len(y), (100, b))
                    ]
                )
            )
            Q = {k: v.copy() for k, v in P.items()}
            ls = []
            for step in range(101):
                ls.append(mlp_loss(Q, xt, yt, False)[0])
                if step == 100:
                    break
                ix = rng.integers(0, len(y), b)
                _, G = mlp_loss(Q, x[ix], y[ix])
                for k in Q:
                    Q[k] -= min(0.8, 0.05 * b) * G[k]
            losses.append(ls)
            params.update({"P_%s_%s" % (b, k): v for k, v in Q.items()})
        return dict(batches=batches, noise=noise, loss=losses, signal=full @ full, **params)

    d = _cache("batch-noise", build, samples=100, updates=100, observations=101)
    return _study(
        "Batch size and gradient noise",
        "Measured gradients of a small MNIST classifier",
        [
            _curve(
                "Noise falls with batch size",
                "Squared distance from the full gradient is averaged over 100 batches.",
                d["noise"],
                ["measured noise"],
                d["batches"],
            ),
            _curve(
                "The inverse-batch reference",
                "Independent sampling predicts variance proportional to 1 / B.",
                [d["noise"], d["noise"][0] / d["batches"]],
                ["measured", "1/B reference"],
                d["batches"],
            ),
            _curve(
                "Linear learning-rate scaling",
                "Learning rates grow with batch size and are capped for stability.",
                d["loss"],
                ["B=%d" % b for b in d["batches"]],
            ),
            _bars(
                "Noise compared with signal",
                "This compares noise magnitude with signal; usefulness is not measured here.",
                [d["signal"], *d["noise"]],
                ["signal", "B=1", "B=4", "B=16", "B=64", "B=256"],
            ),
        ],
    )


def weight_average():
    def build():
        x, y, xt, yt = _digit_data(640)
        rng = np.random.default_rng(17)
        P = mlp_init(49, 24, 10, rng)
        ema = {k: v.copy() for k, v in P.items()}
        swa = {k: v.copy() for k, v in P.items()}
        losses = []
        acc = []
        snaps = []
        count = 0
        for i in range(401):
            if i % 5 == 0:
                ev = [mlp_loss(Q, xt, yt, False) for Q in (P, ema, swa)]
                losses.append([a for a, b in ev])
                acc.append([b for a, b in ev])
                snaps.append(_pack(P))
            if i == 400:
                break
            ix = rng.integers(0, len(y), 32)
            _, G = mlp_loss(P, x[ix], y[ix])
            for k in P:
                P[k] -= 0.8 * G[k]
                ema[k] = 0.97 * ema[k] + 0.03 * P[k]
            if i >= 200:
                count += 1
                for k in P:
                    swa[k] += (P[k] - swa[k]) / count
            else:
                swa = {k: v.copy() for k, v in P.items()}
        curvature = [_sharpness(Q, x[:128], y[:128]) for Q in (P, ema, swa)]
        W = np.array(snaps)
        _, _, vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
        xy = (W - W.mean(0)) @ vt[:2].T
        params = {
            "P_%s_%s" % (label, k): v
            for label, q in (("sgd", P), ("ema", ema), ("swa", swa))
            for k, v in q.items()
        }
        return dict(
            loss=np.array(losses).T, acc=np.array(acc).T, curvature=curvature, xy=xy, **params
        )

    d = _cache("weight-average", build, updates=400, observations=81, ema=0.97)
    return _study(
        "Averaging model weights",
        "MNIST tanh classifier: SGD, EMA and SWA",
        [
            _points(
                "A noisy weight trajectory",
                "PCA projects recorded SGD weights into two dimensions.",
                d["xy"],
            ),
            _curve(
                "Held-out loss",
                "EMA and SWA are evaluated on the same held-out images.",
                d["loss"],
                ["SGD", "EMA", "SWA"],
                np.arange(0, 401, 5),
            ),
            _curve(
                "Held-out accuracy",
                "SWA starts averaging after step 200.",
                d["acc"],
                ["SGD", "EMA", "SWA"],
                np.arange(0, 401, 5),
            ),
            _bars(
                "Curvature at the final weights",
                "Lower curvature is measured, not guaranteed by averaging.",
                d["curvature"],
                ["SGD", "EMA", "SWA"],
            ),
        ],
    )


def precondition():
    def build():
        Q = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
        H = Q @ np.diag([1.0, 100.0]) @ Q.T
        paths = []
        losses = []
        for method in range(3):
            w = np.array([2.0, -1.0])
            m = np.zeros(2)
            v = np.zeros(2)
            ls = []
            path = []
            for t in range(101):
                ls.append(w @ H @ w / 2)
                path.append(w.copy())
                g = H @ w
                if method == 0:
                    step = 0.018 * g
                elif method == 1:
                    m = 0.9 * m + 0.1 * g
                    v = 0.999 * v + 0.001 * g * g
                    step = (
                        0.1
                        * (m / (1 - 0.9 ** (t + 1)))
                        / (np.sqrt(v / (1 - 0.999 ** (t + 1))) + 1e-8)
                    )
                else:
                    step = 0.8 * np.linalg.solve(H, g)
                w -= step
            paths.append(path)
            losses.append(ls)
        x, y, _, _ = _digit_data(128)
        P = mlp_init(49, 12, 10, np.random.default_rng(17))
        _, G = mlp_loss(P, x, y)
        cov = x.T @ x / len(x) + 0.01 * np.eye(49)
        raw = np.linalg.norm(G["W"])
        pre = np.linalg.norm(np.linalg.solve(cov, G["W"]))
        return dict(H=H, paths=paths, loss=losses, cov=cov, norms=[raw, pre])

    d = _cache("precondition", build, steps=101)
    return _study(
        "Preconditioners and correlated directions",
        "Exact quadratic; an MNIST activation-covariance block",
        [
            _matrix(
                "A rotated bowl", "The large off-diagonal entries couple the coordinates.", d["H"]
            ),
            _points(
                "Steps in parameter space",
                "A full curvature block can undo a rotation that a diagonal cannot.",
                d["paths"].reshape(-1, 2),
                np.repeat(np.arange(3), 101),
            ),
            _curve(
                "Objective during optimisation",
                "The block reference uses the exact quadratic Hessian.",
                d["loss"],
                ["GD", "Adam diagonal", "full block"],
            ),
            _matrix(
                "An activation covariance block",
                "An MNIST MLP supplies a real input-covariance preconditioner.",
                d["cov"],
                ("Raw gradient norm %.3g; preconditioned %.3g" % tuple(d["norms"]),),
            ),
        ],
    )


def loss_spikes():
    from . import tinylm

    def build():
        rng = np.random.default_rng(17)
        x = rng.integers(0, 8, (32, 8))
        y = np.roll(x, -1, axis=1)
        cfg = tinylm.Config(vocab=8, width=16, heads=2, kv_heads=2, layers=1, ff=24)
        losses = []
        norms = []
        moments = []
        for beta2, eps, clip in ((0.999, 1e-8, None), (0.9, 1e-8, None), (0.999, 1e-3, 1.0)):
            P = tinylm.init(cfg, seed=17)
            m = {k: np.zeros_like(v) for k, v in P.items()}
            v = {k: np.zeros_like(v) for k, v in P.items()}
            ls = []
            ns = []
            vs = []
            for i in range(141):
                # An abrupt curriculum shift is explicit and reproducible.
                yy = y if i < 60 else (y + 3) % 8
                loss, G = tinylm.loss_grads(P, x, yy, cfg)
                norm = np.sqrt(sum(np.sum(g * g) for g in G.values()))
                ls.append(loss)
                ns.append(norm)
                vs.append(np.sqrt(sum(np.sum(q) for q in v.values())))
                factor = min(1.0, clip / max(norm, 1e-12)) if clip else 1.0
                for k in P:
                    g = G[k] * factor
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g * g
                    P[k] -= (
                        0.025
                        * (m[k] / (1 - 0.9 ** (i + 1)))
                        / (np.sqrt(v[k] / (1 - beta2 ** (i + 1))) + eps)
                    )
            losses.append(ls)
            norms.append(ns)
            moments.append(vs)
        return dict(loss=losses, norm=norms, moment=moments)

    d = _cache("loss-spikes", build, steps=141, shift=60, lr=0.025)
    labels = ["β₂=.999", "β₂=.9", "clip + ε=.001"]
    return _study(
        "Anatomy of a loss spike",
        "Tiny transformer on toy tokens; target rule changes at step 60",
        [
            _curve(
                "The target changes abruptly",
                "A curriculum shift creates a reproducible mismatch in the learned model.",
                d["loss"],
                labels,
            ),
            _curve(
                "Gradient norm",
                "The spike is measured on the same token batch for all three runs.",
                d["norm"],
                labels,
            ),
            _curve(
                "Adam second-moment state",
                "Lower β₂ forgets old gradients faster.",
                d["moment"],
                labels,
            ),
            _bars(
                "Recovery after the shift",
                "Clipping and epsilon changes can trade response speed for stability.",
                d["loss"][:, -10:].mean(1),
                labels,
            ),
        ],
    )


@lru_cache(maxsize=1)
def _word_vectors():
    from . import w2v

    run = w2v.train()
    return run, w2v.unit(run.W_in)


def retrieval():
    from . import gpt2, w2v
    from .gpt2_hooks import run as gpt_run

    passages = (
        "the farmer eats cheese in the village",
        "the baker eats bread in the city",
        "the horse runs to the farm",
        "the doctor drinks milk by the river",
        "the child walks to the school",
    )

    def build():
        model, _ = _word_vectors()
        sent = w2v.sentence_vectors(model, passages + ("the baker eats bread",))
        u = w2v.unit(sent)
        scores = u[:-1] @ u[-1]
        order = np.argsort(-scores)
        prompts = [
            "The baker eats",
            passages[int(order[0])] + ". The baker eats",
            passages[int(order[-1])] + ". The baker eats",
        ]
        ids = gpt2.Tokenizer().encode(" bread")
        assert len(ids) == 1
        probs = []
        tops = []
        for text in prompts:
            pr = gpt2.softmax(gpt_run(text)["logits"][-1])
            probs.append(pr[ids[0]])
            tops.append(np.sort(pr)[-8:][::-1])
        return dict(scores=scores, order=order, probs=probs, tops=tops, similarity=u @ u.T)

    d = _cache("retrieval", build, passages=passages, prompt="The baker eats", answer=" bread")
    return _study(
        "Retrieval-augmented generation",
        "Toy passages, trained word2vec retrieval, real GPT-2 answer probabilities",
        [
            _matrix(
                "Embed question and passages",
                "Cosine similarity compares a question with five short passages.",
                d["similarity"],
            ),
            _bars(
                "Retrieve by cosine",
                "The highest-scoring passage is selected without reading GPT-2 answers.",
                d["scores"],
                ["farmer", "baker", "horse", "doctor", "child"],
            ),
            _bars(
                "Probability of the answer bread",
                "A relevant or irrelevant passage is prepended to the same prompt.",
                d["probs"],
                ["no passage", "top match", "last match"],
            ),
            _matrix(
                "The output distribution changes",
                "Top eight probabilities: no context, retrieved context and low-ranked context.",
                d["tops"],
                ("P(bread): %.3f → %.3f" % tuple(d["probs"][:2]),),
            ),
        ],
        gpt2=True,
    )


class HNSW:
    """Small deterministic hierarchical proximity graph with best-first search.

    It deliberately uses a simple nearest-neighbour construction: every new
    node links to its M nearest earlier nodes at every shared level. Search
    obeys the usual greedy upper layers and bounded best-first base layer.
    """

    def __init__(self, x, m=6, seed=17):
        self.x = np.asarray(x)
        self.m = m
        rng = np.random.default_rng(seed)
        self.levels = np.minimum(5, np.floor(-np.log(rng.random(len(x))) / np.log(m))).astype(int)
        self.edges = [
            {i: set() for i in np.flatnonzero(self.levels >= l)}
            for l in range(int(self.levels.max()) + 1)
        ]
        for _layer, graph in enumerate(self.edges):
            for i in graph:
                earlier = np.array([j for j in graph if j < i], int)
                if not len(earlier):
                    continue
                dist = np.sum((self.x[earlier] - self.x[i]) ** 2, axis=1)
                for j in earlier[np.argsort(dist)[:m]]:
                    graph[i].add(int(j))
                    graph[int(j)].add(i)
        self.entry = int(np.argmax(self.levels))

    def query(self, q, ef=12):
        current = self.entry
        checked = set()
        path = []

        def dist(i):
            checked.add(i)
            return float(np.sum((self.x[i] - q) ** 2))

        current_d = dist(current)
        for graph in self.edges[:0:-1]:
            changed = True
            while changed:
                changed = False
                for j in sorted(graph[current]):
                    d = dist(j)
                    if d < current_d:
                        current, current_d, changed = j, d, True
                        path.append(j)
        import heapq

        candidates = [(current_d, current)]
        best = [(-current_d, current)]
        seen = {current}
        while candidates:
            d, i = heapq.heappop(candidates)
            if len(best) >= ef and d > -best[0][0]:
                break
            path.append(i)
            for j in sorted(self.edges[0][i]):
                if j in seen:
                    continue
                seen.add(j)
                dj = dist(j)
                if len(best) < ef or dj < -best[0][0]:
                    heapq.heappush(candidates, (dj, j))
                    heapq.heappush(best, (-dj, j))
                    if len(best) > ef:
                        heapq.heappop(best)
        answer = min(((-d, i) for d, i in best))[1]
        return answer, len(checked), path


def hnsw():
    def build():
        _, u = _word_vectors()
        graph = HNSW(u)
        rng = np.random.default_rng(17)
        q = u + rng.normal(0, 0.08, u.shape)
        exact = ((q[:, None] - u[None]) ** 2).sum(-1).argmin(1)
        efs = np.array([1, 2, 4, 8, 16, 32])
        recall = []
        work = []
        for ef in efs:
            results = [graph.query(v, int(ef)) for v in q]
            recall.append(np.mean([r[0] == e for r, e in zip(results, exact)]))
            work.append(np.mean([r[1] for r in results]))
        _, _, vt = np.linalg.svd(u - u.mean(0), full_matrices=False)
        xy = (u - u.mean(0)) @ vt[:2].T
        path = graph.query(q[0], 8)[2]
        adj = np.zeros((len(u), len(u)))
        for i, near in graph.edges[0].items():
            adj[i, list(near)] = 1
        return dict(
            xy=xy, levels=graph.levels, adj=adj, efs=efs, recall=recall, work=work, path=xy[path]
        )

    d = _cache("hnsw", build, m=6, ef=[1, 2, 4, 8, 16, 32])
    return _study(
        "Hierarchical vector search",
        "A seeded HNSW-style graph over trained word vectors",
        [
            _points(
                "Sparse upper layers",
                "Random levels produce a hierarchy above the dense base graph.",
                d["xy"],
                d["levels"],
            ),
            _matrix(
                "The base neighbourhood graph",
                "Each word links to nearby previously inserted word vectors.",
                d["adj"],
            ),
            _points(
                "Follow a search path",
                "Greedy descent is followed by a bounded best-first search.",
                d["path"],
            ),
            _curve(
                "Recall and distance evaluations",
                "Increasing the search budget trades more distance calculations for recall.",
                [d["recall"], d["work"] / len(d["xy"])],
                ["recall", "fraction examined"],
                d["efs"],
            ),
        ],
    )


def infonce_loss(W, a, b, temp=0.2, negative_weights=None):
    """Symmetric paired contrastive loss with exact normalization derivatives."""
    za = a @ W
    zb = b @ W
    na = np.maximum(np.linalg.norm(za, axis=1, keepdims=True), 1e-10)
    nb = np.maximum(np.linalg.norm(zb, axis=1, keepdims=True), 1e-10)
    u = za / na
    v = zb / nb
    scores = u @ v.T / temp
    if negative_weights is not None:
        scores = scores + np.log(np.maximum(negative_weights, 1e-20))
    pr = softmax(scores)
    pc = softmax(scores.T)
    loss = float(-0.5 * (np.log(np.diag(pr) + 1e-30).mean() + np.log(np.diag(pc) + 1e-30).mean()))
    grad = (pr + pc.T - 2 * np.eye(len(a))) / (2 * len(a) * temp)
    du = grad @ v
    dv = grad.T @ u
    dza = (du - u * np.sum(u * du, 1, keepdims=True)) / na
    dzb = (dv - v * np.sum(v * dv, 1, keepdims=True)) / nb
    return loss, a.T @ dza + b.T @ dzb, u, v


def _contrastive_run(temp=0.2, mode="random", steps=161):
    x, y, xt, yt = _digit_data(384, 192)
    rng = np.random.default_rng(17)
    W = rng.normal(0, 0.2, (49, 12))
    P = {"W": W}
    opt = Adam(P, 0.004)
    losses = []
    align = []
    uniform = []
    for _ in range(steps):
        ix = rng.integers(0, len(y), 48)
        a = x[ix] + rng.normal(0, 0.05, (48, 49))
        b = x[ix] + rng.normal(0, 0.05, (48, 49))
        weights = None
        if mode != "random":
            weights = np.ones((48, 48))
            if mode == "hard":
                # Different-class lookalikes receive extra weight, never known false negatives.
                diff = y[ix, None] != y[ix][None]
                similarity = a @ b.T
                cutoff = np.quantile(similarity, 0.7, axis=1)[:, None]
                weights[diff & (similarity >= cutoff)] = 5.0
                weights[~diff] = 0.02
            else:
                weights[y[ix, None] == y[ix][None]] = 8.0
            np.fill_diagonal(weights, 1)
        loss, G, _, _ = infonce_loss(P["W"], a, b, temp, weights)
        opt.step(P, {"W": G})
        losses.append(loss)
        z = xt @ P["W"]
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
        sim = z @ z.T
        same = yt[:, None] == yt[None]
        align.append(float(sim[same].mean()))
        uniform.append(float(np.log(np.exp(-2 * np.maximum(0, 2 - 2 * sim)).mean())))
    pred = (
        z @ (x @ P["W"] / np.maximum(np.linalg.norm(x @ P["W"], axis=1, keepdims=True), 1e-10)).T
    ).argmax(1)
    return dict(
        loss=losses,
        alignment=align,
        uniform=uniform,
        z=z,
        acc=np.mean(y[pred] == yt),
        y=yt,
        P_encoder=P["W"],
    )


def temperature():
    def build():
        runs = [_contrastive_run(t) for t in (0.05, 0.2, 1.0)]
        return dict(
            loss=[r["loss"] for r in runs],
            alignment=[r["alignment"] for r in runs],
            uniform=[r["uniform"] for r in runs],
            acc=[r["acc"] for r in runs],
            xy=runs[1]["z"][:, :2],
            classes=runs[1]["y"],
            **{"P_encoder%d" % i: r["P_encoder"] for i, r in enumerate(runs)},
        )

    d = _cache("infonce-temperature", build, temps=[0.05, 0.2, 1.0], steps=161)
    labels = ["τ=.05", "τ=.2", "τ=1"]
    return _study(
        "Temperature in contrastive learning",
        "Linear encoders learn paired noisy MNIST views",
        [
            _curve(
                "The contrastive objective",
                "Temperature rescales similarity before softmax.",
                d["loss"],
                labels,
            ),
            _points(
                "A two-coordinate view",
                "These are two learned coordinates, not a new projection fit.",
                d["xy"],
                d["classes"],
            ),
            _curve(
                "Measured alignment",
                "Mean cosine of same-label test images measures class alignment.",
                d["alignment"],
                labels,
            ),
            _bars(
                "Held-out nearest-neighbour accuracy",
                "The temperature sweep is evaluated on the same test images.",
                d["acc"],
                labels,
            ),
        ],
        mnist=True,
    )


def hard_negatives():
    def build():
        runs = [_contrastive_run(0.2, mode) for mode in ("random", "hard", "false")]
        return dict(
            loss=[r["loss"] for r in runs],
            alignment=[r["alignment"] for r in runs],
            uniform=[r["uniform"] for r in runs],
            acc=[r["acc"] for r in runs],
            **{"P_encoder%d" % i: r["P_encoder"] for i, r in enumerate(runs)},
        )

    d = _cache("hard-negatives", build, steps=161, modes=["random", "hard", "false"])
    names = ["random", "hard different class", "false negatives"]
    return _study(
        "Choosing negative examples",
        "Measured contrastive training on noisy MNIST image pairs",
        [
            _curve(
                "The same paired-view task",
                "Negative weights change which competing examples dominate.",
                d["loss"],
                names,
            ),
            _curve(
                "Within-class alignment",
                "False negatives explicitly push images of the same digit apart.",
                d["alignment"],
                names,
            ),
            _curve(
                "Embedding uniformity",
                "Log mean exp(−2 squared distance) measures crowding on the sphere.",
                d["uniform"],
                names,
            ),
            _bars(
                "Held-out retrieval accuracy",
                "Hard negatives are not assumed to win; compare the observed accuracies.",
                d["acc"],
                names,
            ),
        ],
        mnist=True,
    )


def embedding_width():
    from . import w2v

    def build():
        words = w2v.vocabulary()
        index = {w: i for i, w in enumerate(words)}
        sents = w2v.corpus(1600, seed=17)
        pairs = w2v.pairs(sents, index)
        rng = np.random.default_rng(17)
        pairs = pairs[rng.permutation(len(pairs))]
        dims = np.array([2, 4, 8, 16, 32, 64, 128, 256])
        scores = []
        nested = []
        losses = []
        for dim in dims:
            W = rng.normal(0, 0.05, (len(words), dim))
            V = rng.normal(0, 0.05, (len(words), dim))
            ls = []
            for _step in range(601):
                ids = rng.integers(0, len(pairs), 128)
                c, o = pairs[ids].T
                neg = rng.integers(0, len(words), (128, 5))
                loss, ri, gi, ro, go = w2v.sgns_grads(W, V, c, o, neg)
                np.add.at(W, ri, -0.8 * gi)
                np.add.at(V, ro, -0.8 * go)
                ls.append(loss)
            same, diff = w2v.category_cosines(W, words)
            scores.append(same - diff)
            losses.append(ls)
            if dim == 256:
                # Nested objective: explicitly fine-tune prefixes rather than truncate blindly.
                for _ in range(501):
                    ids = rng.integers(0, len(pairs), 128)
                    c, o = pairs[ids].T
                    neg = rng.integers(0, len(words), (128, 5))
                    for d in (4, 16, 64, 256):
                        _, ri, gi, ro, go = w2v.sgns_grads(W[:, :d], V[:, :d], c, o, neg)
                        np.add.at(W[:, :d], ri, -0.15 * gi)
                        np.add.at(V[:, :d], ro, -0.15 * go)
                for d in dims:
                    a, b = w2v.category_cosines(W[:, :d], words)
                    nested.append(a - b)
        return dict(dims=dims, scores=scores, nested=nested, loss=losses)

    d = _cache("embedding-width", build, steps=601, nested_steps=501, corpus=1600)
    return _study(
        "How wide should an embedding be?",
        "Skip-gram trained on the generated word2vec corpus",
        [
            _curve(
                "Eight trained widths",
                "Every width trains on the same corpus and sampled skip-gram objective.",
                d["loss"][[0, 3, 7]],
                ["2 dimensions", "16 dimensions", "256 dimensions"],
            ),
            _curve(
                "Category separation",
                "Quality is within-category cosine minus between-category cosine.",
                d["scores"],
                ["independently trained"],
                d["dims"],
            ),
            _curve(
                "Nested embedding prefixes",
                "One wide model is fine-tuned with losses on nested coordinate prefixes.",
                [d["scores"], d["nested"]],
                ["separate models", "nested prefixes"],
                d["dims"],
            ),
            _bars(
                "Storage per word",
                "Float32 coordinate storage grows directly with width.",
                d["dims"] * 4,
                ["d=%d" % v for v in d["dims"]],
            ),
        ],
    )


def allreduce():
    def build():
        rng = np.random.default_rng(17)
        images, labels, _, _ = _digit_data(64)
        x = images[:, :12]
        y = np.eye(10)[labels][:, :8]
        w = rng.normal(size=(12, 8))

        def grad(a, b):
            return a.T @ (a @ w - b) / len(a)

        gs = np.stack([grad(a, b) for a, b in zip(np.split(x, 4), np.split(y, 4))])
        full = grad(x, y)
        average = gs.mean(0)
        # Simulate reduce-scatter, then all-gather, by passing owned chunks.
        chunks = np.stack([np.split(g.ravel(), 4) for g in gs])
        reduced = chunks.copy()
        reduction_history = [reduced.copy()]
        for turn in range(3):
            before = reduced.copy()
            for worker in range(4):
                chunk = (worker - turn) % 4
                destination = (worker + 1) % 4
                reduced[destination, chunk] += before[worker, chunk]
            reduction_history.append(reduced.copy())
        # Rank r now owns reduced chunk (r+1) mod P.
        reduced /= 4
        owned = np.zeros_like(reduced)
        for worker in range(4):
            owned[worker, (worker + 1) % 4] = reduced[worker, (worker + 1) % 4]
        history = [owned.copy()]
        for turn in range(3):
            before = owned.copy()
            for worker in range(4):
                chunk = (worker + 1 - turn) % 4
                owned[(worker + 1) % 4, chunk] = before[worker, chunk]
            history.append(owned.copy())
        return dict(
            local=gs.reshape(4, -1),
            full=full,
            average=average,
            ring=history,
            reduce_ring=reduction_history,
            bytes=np.arange(7) * full.nbytes / 4 * 4,
        )

    d = _cache("allreduce", build, workers=4, shape=[12, 8], batch=64, ring_version=2)
    err = float(np.max(np.abs(d["full"] - d["average"])))
    return _study(
        "Data parallelism and ring all-reduce",
        "Actual MNIST squared-error gradients; simulated four-worker communication",
        [
            _matrix(
                "Different batches, different gradients",
                "Each worker owns the same weights and sees sixteen examples.",
                d["local"],
            ),
            _matrix(
                "Reduce-scatter",
                "Three neighbour exchanges accumulate a different chunk at each worker.",
                d["reduce_ring"][-1].reshape(4, -1),
            ),
            _matrix(
                "All-gather",
                "Three neighbour exchanges distribute all reduced chunks.",
                d["ring"][-1].reshape(4, -1),
            ),
            _bars(
                "Communication per worker",
                "Reduce-scatter plus all-gather moves 2(P−1)/P gradient bytes.",
                [d["full"].nbytes, 2 * 3 / 4 * d["full"].nbytes],
                ["gradient size", "ring bytes"],
                ("Mean gradient error: %.2g" % err,),
            ),
        ],
    )


def tensor_pipeline():
    def build():
        rng = np.random.default_rng(17)
        x = _digit_data(16)[0][:, :12]
        w = rng.normal(size=(12, 16))
        cols = np.concatenate([x @ a for a in np.split(w, 4, axis=1)], axis=1)
        rows = sum(a @ b for a, b in zip(np.split(x, 4, axis=1), np.split(w, 4, axis=0)))
        timeline = np.zeros((4, 11))
        for stage in range(4):
            for micro in range(8):
                timeline[stage, stage + micro] = micro + 1
        ms = np.array([1, 2, 4, 8, 16, 32])
        utilization = ms / (ms + 3)
        return dict(
            x=x,
            w=w,
            cols=cols,
            rows=rows,
            reference=x @ w,
            timeline=timeline,
            micro=ms,
            util=utilization,
        )

    d = _cache("tensor-pipeline", build, workers=4, micro=8)
    err = max(np.max(abs(d["cols"] - d["reference"])), np.max(abs(d["rows"] - d["reference"])))
    return _study(
        "Tensor and pipeline parallelism",
        "MNIST input matrix; exact partitioning and an idealized pipeline schedule",
        [
            _matrix(
                "Column partitions concatenate",
                "Four workers compute disjoint output columns.",
                d["cols"],
            ),
            _matrix(
                "Row partitions reduce",
                "Four partial matrix products sum to the same output.",
                d["rows"],
                ("Maximum numerical error: %.2g" % err,),
            ),
            _matrix(
                "Micro-batches fill a pipeline",
                "Rows are four stages; colours identify eight micro-batches.",
                d["timeline"],
            ),
            _curve(
                "The pipeline bubble shrinks",
                "With P stages and M micro-batches, utilisation is M/(M+P−1).",
                d["util"],
                ["utilisation"],
                d["micro"],
            ),
        ],
    )


def zero_sharding():
    n = 7_000_000_000
    workers = 8
    fp = 2
    grad = 2
    opt = 12
    # Mixed precision: FP16 weights/gradients, FP32 master plus Adam moments.
    memory = (
        np.array(
            [
                n * (fp + grad + opt),
                n * (fp + grad + opt / workers),
                n * (fp + (grad + opt) / workers),
                n * (fp + grad + opt) / workers,
            ]
        )
        / 1e9
    )
    parts = (
        np.array(
            [
                [fp, grad, opt],
                [fp, grad, opt / workers],
                [fp, grad / workers, opt / workers],
                [fp / workers, grad / workers, opt / workers],
            ]
        )
        * n
        / 1e9
    )
    return _study(
        "ZeRO: sharding training state",
        "Exact storage arithmetic; 7 billion parameters across eight workers",
        [
            _bars(
                "Where the bytes go",
                "Mixed precision includes a float32 master copy and both Adam moments.",
                n * np.array([fp, grad, opt]) / 1e9,
                ["weights", "gradients", "optimizer GB"],
            ),
            _matrix(
                "Progressively shard the state",
                "Rows: replicated and stages 1–3. Columns: weights, gradients, optimizer state.",
                parts,
            ),
            _bars(
                "Gigabytes per worker",
                "Activations, temporary buffers and communication storage are excluded.",
                memory,
                ["replicated", "stage 1", "stage 2", "stage 3"],
            ),
            _curve(
                "Scaling with worker count",
                "Stage 3 divides model state; replicated training stores every copy.",
                [
                    np.full(6, n * (fp + grad + opt) / 1e9),
                    n * (fp + grad + opt) / 2 ** np.arange(6) / 1e9,
                ],
                ["replicated", "stage 3"],
                2 ** np.arange(6),
            ),
        ],
    )


def checkpoint_grad(W, x, y, spacing):
    """An actual recomputed backward pass, storing only segment boundaries."""
    n = len(W)
    cache = {0: x}
    a = x
    for i, w in enumerate(W):
        a = np.tanh(a @ w)
        if (i + 1) % spacing == 0 or i + 1 == n:
            cache[i + 1] = a
    grad = (a - y) / a.size
    out = [None] * n
    recompute = 0
    boundaries = sorted(cache)
    for start, end in zip(boundaries[-2::-1], boundaries[:0:-1]):
        acts = [cache[start]]
        for i in range(start, end):
            acts.append(np.tanh(acts[-1] @ W[i]))
            recompute += 1
        for i in range(end - 1, start - 1, -1):
            delta = grad * (1 - acts[i - start + 1] ** 2)
            out[i] = acts[i - start].T @ delta
            grad = delta @ W[i].T
    return out, len(cache), recompute


def checkpointing():
    def build():
        rng = np.random.default_rng(17)
        W = [rng.normal(0, 0.22, (16, 16)) for _ in range(16)]
        x = rng.normal(size=(8, 16))
        y = rng.normal(size=(8, 16))
        ref, _, _ = checkpoint_grad(W, x, y, 1)
        spacings = np.array([1, 2, 4, 8, 16])
        errors = []
        mem = []
        work = []
        for s in spacings:
            g, c, r = checkpoint_grad(W, x, y, int(s))
            errors.append(max(np.max(abs(a - b)) for a, b in zip(g, ref)))
            mem.append(c + int(s))
            work.append(16 + r)
        return dict(spacings=spacings, errors=errors, mem=mem, work=work, gradient=ref[0])

    d = _cache("checkpointing", build, layers=16, width=16)
    return _study(
        "Activation checkpointing",
        "A sixteen-layer tanh network with recomputed backward segments",
        [
            _bars(
                "Store boundaries, not every layer",
                "Peak activation slots include segment boundaries and the active segment.",
                d["mem"],
                ["k=%d" % i for i in d["spacings"]],
            ),
            _bars(
                "Recompute during backward",
                "This simple scheme recomputes each segment once.",
                d["work"],
                ["k=%d" % i for i in d["spacings"]],
            ),
            _matrix(
                "The actual gradient",
                "A seeded network supplies a concrete first-layer weight gradient.",
                d["gradient"],
            ),
            _bars(
                "Same gradients across spacings",
                "The measured difference from storing every activation is numerical precision.",
                d["errors"],
                ["k=%d" % i for i in d["spacings"]],
            ),
        ],
    )


def roofline():
    def build():
        rng = np.random.default_rng(17)
        n = 256
        x = rng.normal(size=(n, n))
        w = rng.normal(size=(n, n))
        q = rng.normal(size=(128, 64))
        k = q.copy()
        v = q.copy()
        ops = [
            lambda: x @ w,
            lambda: softmax(q @ k.T / 8) @ v,
            lambda: (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5),
            lambda: x[:1] @ w,
        ]
        flops = np.array([2 * n**3, 4 * 128**2 * 64, 8 * n * n, 2 * n * n], float)
        byte = np.array([3 * n * n, 3 * 128 * 64 + 2 * 128**2, 2 * n * n, n * n + 2 * n]) * 8
        secs = []
        for op in ops:
            op()
            samples = []
            for _ in range(9):
                t = time.perf_counter()
                op()
                samples.append(time.perf_counter() - t)
            secs.append(np.median(samples))
        return dict(
            flops=flops,
            bytes=byte,
            seconds=secs,
            intensity=flops / byte,
            throughput=flops / np.array(secs) / 1e9,
        )

    d = _cache("roofline", build, n=256, dtype="float64", repeats=9)
    names = ["matmul", "attention", "layer norm", "decode"]
    # The roof is an empirical lower-bound envelope, not a claimed chip spec.
    bw = max(d["bytes"] / d["seconds"]) / 1e9
    peak = max(d["throughput"])
    xs = np.geomspace(0.01, 100, 80)
    roof = np.minimum(peak, bw * xs)
    return _study(
        "The roofline performance model",
        "Modeled dominant FLOPs/logical bytes; measured local NumPy timings",
        [
            _bars(
                "Arithmetic intensity",
                "FLOPs per input/output byte, including stored attention intermediates.",
                d["intensity"],
                names,
            ),
            _bars(
                "Observed throughput",
                "Nine warmed calls per operation; median timing, on this machine.",
                d["throughput"],
                names,
            ),
            _curve(
                "An empirical envelope",
                "This fitted envelope is a model, not a measured chip specification.",
                roof,
                ["local envelope"],
                xs,
                ("Peak observed: %.2f GFLOP/s" % peak,),
            ),
            _bars(
                "Latency of each operation",
                "Small decoding matmuls may move many bytes for little arithmetic.",
                d["seconds"] * 1e6,
                names,
                ("Times are local observations, in microseconds.",),
            ),
        ],
    )


def bandits():
    def build():
        means = np.linspace(0.1, 0.85, 10)
        steps = 2000
        regrets = []
        counts = []
        estimates = []
        for method in range(3):
            all_regret = []
            last_count = []
            last_q = []
            for seed in range(12):
                r = np.random.default_rng(1700 + seed)
                n = np.zeros(10)
                wins = np.zeros(10)
                cumulative = 0.0
                regret = []
                for t in range(steps):
                    q = wins / np.maximum(n, 1)
                    if method == 0:
                        a = int(r.integers(10)) if r.random() < 0.1 else int(q.argmax())
                    elif method == 1:
                        a = int(np.argmax(q + np.sqrt(2 * np.log(t + 2) / np.maximum(n, 1))))
                        if np.any(n == 0):
                            a = int(np.flatnonzero(n == 0)[0])
                    else:
                        a = int(np.argmax(r.beta(wins + 1, n - wins + 1)))
                    reward = r.random() < means[a]
                    n[a] += 1
                    wins[a] += reward
                    cumulative += means.max() - means[a]
                    regret.append(cumulative)
                all_regret.append(regret)
                last_count.append(n)
                last_q.append(wins / np.maximum(n, 1))
            regrets.append(np.mean(all_regret, 0))
            counts.append(np.mean(last_count, 0))
            estimates.append(np.mean(last_q, 0))
        return dict(means=means, regret=regrets, counts=counts, q=estimates)

    d = _cache("bandits", build, arms=10, steps=2000, seeds=12)
    return _study(
        "Exploration in a ten-armed bandit",
        "Seeded Bernoulli rewards; averages over twelve independent runs",
        [
            _bars(
                "Unknown reward probabilities",
                "The simulator knows the probabilities; each learner starts without them.",
                d["means"],
                ["arm %d" % i for i in range(10)],
            ),
            _curve(
                "Accumulated expected regret",
                "Each choice costs the gap between its mean reward and the best arm.",
                d["regret"],
                ["ε-greedy", "UCB", "Thompson"],
            ),
            _matrix(
                "Where exploration is spent",
                "Rows show average counts for ε-greedy, UCB and Thompson sampling.",
                d["counts"],
            ),
            _matrix(
                "Learned reward estimates",
                "The posterior mean or empirical mean is compared after 2,000 pulls.",
                d["q"],
            ),
        ],
    )


def policy_train(baseline=True, actor_critic=False, episodes=1800, seed=17):
    from .gridworld import World

    world = World()
    rng = np.random.default_rng(seed)
    logits = np.zeros((world.h, world.w, 4))
    value = np.zeros((world.h, world.w))
    lengths = []
    returns = []
    gradvar = []
    values = []
    advantages = []
    snaps = []
    gamma = 0.97
    free = [tuple(a) for a in np.argwhere(~world.walls) if not world.terminal(tuple(a))]
    for episode in range(episodes):
        s = world.start if episode % 3 else free[int(rng.integers(len(free)))]
        trajectory = []
        total = 0.0
        for _step in range(150):
            p = softmax(logits[s])
            entropy = -np.sum(p * np.log(p + 1e-30))
            logits[s] += 0.01 * p * (-np.log(p + 1e-30) - entropy)
            a = int(rng.choice(4, p=p))
            nxt, r = world.step(s, a)
            terminal = world.terminal(nxt)
            if actor_critic:
                delta = r + gamma * (0.0 if terminal else value[nxt]) - value[s]
                g = -p
                g[a] += 1
                logits[s] += 0.12 * delta * g
                value[s] += 0.15 * delta
                advantages.append(delta)
            trajectory.append((s, a, r, p.copy()))
            total += r
            s = nxt
            if terminal:
                break
        G = 0.0
        gradients = []
        if not actor_critic:
            for s, a, r, p in trajectory[::-1]:
                G = r + gamma * G
                adv = G - value[s] if baseline else G
                g = -p
                g[a] += 1
                gradients.append(adv * g)
                logits[s] += 0.10 * adv * g
                if baseline:
                    value[s] += 0.10 * (G - value[s])
        else:
            gradients = [p * (r - value[s]) for s, a, r, p in trajectory]
        lengths.append(len(trajectory))
        returns.append(total)
        gradvar.append(np.mean(np.var(np.array(gradients), axis=0)))
        if episode % 30 == 0:
            values.append(value.copy())
            snaps.append(softmax(logits).copy())
    return dict(
        lengths=lengths,
        returns=returns,
        gradvar=gradvar,
        value=value,
        policy=softmax(logits),
        values=values,
        snaps=snaps,
        advantages=advantages,
    )


def _rolling(x, n=30):
    return np.convolve(x, np.ones(n) / n, "valid")


def policy_gradient():
    def build():
        a = policy_train(False)
        b = policy_train(True)
        return dict(
            returns=[_rolling(a["returns"]), _rolling(b["returns"])],
            variance=[_rolling(a["gradvar"]), _rolling(b["gradvar"])],
            policy=b["policy"],
            value=b["value"],
            lengths=[_rolling(a["lengths"]), _rolling(b["lengths"])],
        )

    d = _cache("policy-gradient", build, episodes=1800, exploring_starts=True, entropy=0.01)
    return _study(
        "REINFORCE and a baseline",
        "Existing gridworld; exploring starts every third training episode",
        [
            _matrix(
                "A learned stochastic policy",
                "Each grid cell has four probabilities, ordered up, right, down, left.",
                d["policy"].reshape(48, 4),
            ),
            _curve(
                "Measured episode returns",
                "Both agents learn only from sampled complete episodes.",
                d["returns"],
                ["no baseline", "learned baseline"],
            ),
            _curve(
                "Policy-update variance",
                "Subtracting a state-value baseline changes the sampled gradient variance.",
                d["variance"],
                ["no baseline", "learned baseline"],
            ),
            _curve(
                "Steps until termination",
                "Curves are thirty-episode rolling means; shorter is not always successful.",
                d["lengths"],
                ["no baseline", "learned baseline"],
            ),
        ],
    )


def actor_critic():
    def build():
        r = policy_train(True, True)
        return dict(
            value=r["value"],
            policy=r["policy"],
            returns=_rolling(r["returns"]),
            lengths=_rolling(r["lengths"]),
            adv=np.array(r["advantages"])[np.abs(r["advantages"]) > 1e-10][:250],
            values=r["values"],
        )

    d = _cache("actor-critic", build, episodes=1800, exploring_starts=True, entropy=0.01)
    return _study(
        "The actor and its critic",
        "Existing gridworld; exploring starts every third training episode",
        [
            _matrix(
                "The critic estimates value",
                "Temporal-difference updates bootstrap from the next state.",
                d["value"],
            ),
            _matrix(
                "The actor chooses actions",
                "Positive advantage increases the sampled action probability.",
                d["policy"].reshape(48, 4),
            ),
            _curve(
                "One-step advantages",
                "δ = reward + γ V(next) − V(now) drives both updates.",
                d["adv"],
                ["TD advantage"],
            ),
            _curve(
                "Episode returns",
                "The first episode begins with a uniform policy and zero value estimates.",
                d["returns"],
                ["actor-critic"],
            ),
        ],
    )


def qnet_init(rng):
    return dict(
        W=rng.normal(0, 0.5, (2, 24)), b=np.zeros(24), U=rng.normal(0, 0.2, (24, 4)), v=np.zeros(4)
    )


def qnet_forward(P, x):
    h = np.tanh(x @ P["W"] + P["b"])
    return h @ P["U"] + P["v"], h


def qnet_loss(P, x, a, target):
    q, h = qnet_forward(P, x)
    error = q[np.arange(len(a)), a] - target
    z = np.zeros_like(q)
    z[np.arange(len(a)), a] = error / len(a)
    dh = (z @ P["U"].T) * (1 - h * h)
    return float(0.5 * np.mean(error**2)), dict(W=x.T @ dh, b=dh.sum(0), U=h.T @ z, v=z.sum(0))


def dqn():
    def step(s, a, rng):
        delta = np.array([[0, 0.18], [0.18, 0], [0, -0.18], [-0.18, 0]])[a]
        ns = np.clip(s + delta + rng.normal(0, 0.015, 2), 0, 1)
        done = np.linalg.norm(ns - np.array([0.9, 0.9])) < 0.16
        return ns, 1.0 if done else -0.01, done

    def build():
        result = []
        for replay in (False, True):
            rng = np.random.default_rng(17)
            P = qnet_init(rng)
            target = {k: v.copy() for k, v in P.items()}
            opt = Adam(P, 0.003)
            buffer = []
            returns = []
            losses = []
            s = np.array([0.1, 0.1])
            ret = 0.0
            length = 0
            for t in range(6000):
                q, _ = qnet_forward(P, s[None])
                eps = max(0.1, 1 - t / 4000)
                a = int(rng.integers(4)) if rng.random() < eps else int(q[0].argmax())
                ns, r, done = step(s, a, rng)
                buffer.append((s.copy(), a, r, ns.copy(), done))
                if len(buffer) > 2000:
                    buffer.pop(0)
                batch = (
                    [buffer[j] for j in rng.integers(0, len(buffer), 32)]
                    if replay
                    else [buffer[-1]]
                )
                xx = np.array([b[0] for b in batch])
                aa = np.array([b[1] for b in batch])
                rr = np.array([b[2] for b in batch])
                nn = np.array([b[3] for b in batch])
                dd = np.array([b[4] for b in batch])
                qq, _ = qnet_forward(target if replay else P, nn)
                yy = rr + 0.96 * (~dd) * qq.max(1)
                loss, G = qnet_loss(P, xx, aa, yy)
                opt.step(P, G)
                losses.append(loss)
                if replay and t % 100 == 0:
                    target = {k: v.copy() for k, v in P.items()}
                ret += r
                length += 1
                s = ns
                if done or length == 80:
                    returns.append(ret)
                    ret = 0.0
                    length = 0
                    s = np.array([0.1, 0.1])
            grid = np.array([(a, b) for a in np.linspace(0, 1, 12) for b in np.linspace(0, 1, 12)])
            q, _ = qnet_forward(P, grid)
            result.append(dict(loss=np.array(losses).reshape(-1, 50).mean(1), ret=returns, q=q))
        n = min(len(r["ret"]) for r in result)
        return dict(
            loss=[r["loss"] for r in result],
            returns=[r["ret"][:n] for r in result],
            q=result[1]["q"],
            buffer=np.array([b[0] for b in buffer[-144:]]),
        )

    d = _cache("dqn", build, steps=6000, hidden=24, target_period=100)
    return _study(
        "Replay and a target network",
        "Small DQN in a continuous two-coordinate gridworld",
        [
            _points(
                "Remember past experience",
                "A replay buffer stores transitions from earlier policies.",
                d["buffer"],
            ),
            _curve(
                "Temporal-difference loss",
                "The target network refreshes every one hundred environment steps.",
                d["loss"],
                ["online only", "replay + target"],
                np.arange(120) * 50,
            ),
            _curve(
                "Measured episode return",
                "Compare the measured stability and success of both training runs.",
                d["returns"],
                ["online only", "replay + target"],
            ),
            _matrix(
                "Learned Q-values",
                "Rows enumerate a twelve-by-twelve position grid; columns are four actions.",
                d["q"],
            ),
        ],
    )


def tictactoe_winner(board):
    b = np.asarray(board).reshape(3, 3)
    lines = [*b, *b.T, np.diag(b), np.diag(np.fliplr(b))]
    for line in lines:
        if abs(int(np.sum(line))) == 3:
            return int(line[0])
    return 0 if np.any(b == 0) else 2


class MCTS:
    def __init__(self, board, player=1, seed=17):
        self.root = tuple(board)
        self.player = player
        self.rng = np.random.default_rng(seed)
        self.n = {}
        self.v = {}
        self.children = {}

    def simulate(self):
        board = self.root
        player = self.player
        path = []
        while True:
            key = (board, player)
            path.append(key)
            winner = tictactoe_winner(board)
            if winner:
                break
            moves = np.flatnonzero(np.array(board) == 0)
            if key not in self.children:
                self.children[key] = []
                for a in moves:
                    b = list(board)
                    b[a] = player
                    self.children[key].append((tuple(b), -player))
                child = self.children[key][int(self.rng.integers(len(moves)))]
                board, player = child
                path.append(child)
                # Uniform random rollout from the newly expanded leaf.
                while not (winner := tictactoe_winner(board)):
                    a = int(self.rng.choice(np.flatnonzero(np.array(board) == 0)))
                    b = list(board)
                    b[a] = player
                    board = tuple(b)
                    player = -player
                break
            children = self.children[key]
            unvisited = [c for c in children if self.n.get(c, 0) == 0]
            if unvisited:
                board, player = unvisited[int(self.rng.integers(len(unvisited)))]
                continue
            sign = 1 if player == self.player else -1

            def ucb(c, sign=sign, key=key):
                return sign * self.v[c] / self.n[c] + np.sqrt(
                    2 * np.log(max(self.n.get(key, 1), 1)) / self.n[c]
                )

            board, player = max(children, key=ucb)
        reward = 0.0 if winner == 2 else float(winner == self.player) * 2 - 1
        for key in path:
            self.n[key] = self.n.get(key, 0) + 1
            self.v[key] = self.v.get(key, 0) + reward


def mcts():
    def build():
        board = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        tree = MCTS(board)
        moves = np.flatnonzero(np.array(board) == 0)
        counts = []
        values = []
        for _ in range(501):
            tree.simulate()
            ns = []
            vs = []
            for a in moves:
                b = list(board)
                b[a] = 1
                k = (tuple(b), -1)
                ns.append(tree.n.get(k, 0))
                vs.append(tree.v.get(k, 0) / max(tree.n.get(k, 0), 1))
            counts.append(ns)
            values.append(vs)
        return dict(
            board=np.array(board).reshape(3, 3),
            moves=moves,
            counts=np.array(counts).T,
            values=np.array(values).T,
        )

    d = _cache("mcts", build, simulations=501)
    labels = ["cell %d" % v for v in d["moves"]]
    return _study(
        "Monte Carlo tree search",
        "Tic-tac-toe: UCB selection, expansion, random rollout and backup",
        [
            _matrix(
                "Find the winning move",
                "X is +1, O is −1; X can complete the top row immediately.",
                d["board"],
            ),
            _curve(
                "Visits flow toward promising moves",
                "The tree balances exploitation with UCB exploration.",
                d["counts"],
                labels,
            ),
            _curve(
                "Backed-up values",
                "Terminal wins, draws and losses are +1, 0 and −1 from the root player.",
                d["values"],
                labels,
            ),
            _bars(
                "Choose the most visited move",
                "Each bar is a legal move after five hundred and one simulations.",
                d["counts"][:, -1],
                labels,
            ),
        ],
    )


def reward_shaping():
    from .gridworld import World

    def build():
        world = World()
        curves = []
        paths = []
        qtables = []

        def potential(s):
            return -0.1 * (abs(world.goal[0] - s[0]) + abs(world.goal[1] - s[1]))

        for mode in range(3):
            rng = np.random.default_rng(17)
            Q = np.zeros((world.h, world.w, 4))
            returns = []
            free = [tuple(a) for a in np.argwhere(~world.walls) if not world.terminal(tuple(a))]
            for e in range(1600):
                s = world.start if e % 3 else free[int(rng.integers(len(free)))]
                total = 0.0
                for _ in range(160):
                    a = (
                        int(rng.integers(4))
                        if rng.random() < max(0.08, 0.995**e)
                        else int(Q[s].argmax())
                    )
                    ns, r = world.step(s, a)
                    done = world.terminal(ns)
                    shaped = r
                    if mode == 1:
                        shaped += 0.95 * (0.0 if done else potential(ns)) - potential(s)
                    if mode == 2:
                        shaped += 0.1 * (not done)  # survival bonus can reward endless loops
                    target = shaped + 0.95 * (0 if done else Q[ns].max())
                    Q[s][a] += 0.4 * (target - Q[s][a])
                    total += r
                    s = ns
                    if done:
                        break
                returns.append(total)
            path = world.greedy_path(Q, 80)
            curves.append(_rolling(returns))
            paths.append(path)
            qtables.append(Q.max(-1))
        padded = np.full((3, 81, 2), np.nan)
        for i, p in enumerate(paths):
            padded[i, : len(p)] = p
        return dict(
            returns=curves, paths=padded, values=qtables, lengths=[len(p) - 1 for p in paths]
        )

    d = _cache("reward-shaping", build, episodes=1600, gamma=0.95, exploring_starts=True)
    names = ["sparse", "potential-based", "survival bonus"]
    return _study(
        "Rewards can change the task",
        "Existing gridworld; exploring starts every third training episode",
        [
            _curve(
                "Measure the original task reward",
                "All curves report goal or penalty reward before shaping.",
                d["returns"],
                names,
            ),
            _matrix(
                "A useful potential",
                "γ Φ(next) − Φ(now) telescopes and preserves discounted optimal policies.",
                d["values"][1],
            ),
            _matrix(
                "A bonus for staying alive",
                "Adding a reward every nonterminal step can make loops valuable.",
                d["values"][2],
            ),
            _bars(
                "Greedy rollout length",
                "Rollouts stop at a terminal state or after eighty moves.",
                d["lengths"],
                names,
            ),
        ],
    )


def hopfield():
    def build():
        from .mnist import load

        x, y, _, _ = load()
        ids = [int(np.flatnonzero(y == i)[0]) for i in (0, 1, 2)]
        patterns = np.where(x[ids] > 0.3, 1.0, -1.0)
        W = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(W, 0)
        rng = np.random.default_rng(17)
        state = patterns[1].copy()
        state[rng.random(784) < 0.24] *= -1
        snapshots = [state.copy()]
        energy = [-0.5 * state @ W @ state]
        for _ in range(12):
            for j in rng.permutation(784):
                field = W[j] @ state
                if field != 0:
                    state[j] = 1 if field > 0 else -1
            snapshots.append(state.copy())
            energy.append(-0.5 * state @ W @ state)
        loads = []
        overlaps = []
        for count in (1, 3, 8, 16, 32):
            pp = np.where(x[:count] > 0.3, 1.0, -1.0)
            ww = pp.T @ pp / 784
            np.fill_diagonal(ww, 0)
            ss = pp[0].copy()
            ss[rng.random(784) < 0.24] *= -1
            for _ in range(6):
                for j in rng.permutation(784):
                    ss[j] = 1 if ww[j] @ ss >= 0 else -1
            loads.append(count)
            overlaps.append(float(ss @ pp[0] / 784))
        # Modern Hopfield retrieval uses softmax attention over stored patterns.
        prob = softmax(patterns @ snapshots[0] / 28)
        return dict(
            patterns=patterns.reshape(3, 28, 28),
            snapshots=np.array(snapshots).reshape(-1, 28, 28),
            energy=energy,
            loads=loads,
            overlap=overlaps,
            prob=prob,
        )

    d = _cache("hopfield", build, patterns=[0, 1, 2], flip=0.24, sweeps=12)
    return _study(
        "Hopfield memory and attractors",
        "Real MNIST digits; asynchronous binary updates and symmetric weights",
        [
            _images(
                "Store three memories",
                "The Hebbian weight matrix sums three outer products and clears its diagonal.",
                (d["patterns"] + 1) / 2,
                ["zero", "one", "two"],
            ),
            _images(
                "Evolve a corrupted digit",
                "Asynchronous updates lower energy; exact memory recovery is not guaranteed.",
                (d["snapshots"][[0, 1, 3, 12]] + 1) / 2,
                ["corrupt", "sweep 1", "sweep 3", "sweep 12"],
            ),
            _curve(
                "Energy cannot increase",
                "Each asynchronous sign update minimizes one coordinate of the energy.",
                d["energy"],
                ["energy"],
            ),
            _curve(
                "Measure recall as memory load changes",
                "Overlap with the initial memory is measured separately for each pattern count.",
                d["overlap"],
                ["memory overlap"],
                d["loads"],
                notes=("Modern attention weights: %.2f, %.2f, %.2f" % tuple(d["prob"]),),
            ),
        ],
        mnist=True,
    )


def _nca_filters(state):
    mean = (
        np.roll(state, 1, 0) + np.roll(state, -1, 0) + np.roll(state, 1, 1) + np.roll(state, -1, 1)
    ) / 4
    dx = (np.roll(state, -1, 1) - np.roll(state, 1, 1)) / 2
    dy = (np.roll(state, -1, 0) - np.roll(state, 1, 0)) / 2
    return mean, dx, dy


def nca_rollout(P, state, seedmap, steps, keep=False):
    records = []
    for _ in range(steps):
        mean, dx, dy = _nca_filters(state)
        f = np.concatenate([state, mean, dx, dy, seedmap[..., None]], axis=-1)
        h = np.tanh(f @ P["A"] + P["b"])
        nxt = np.tanh(0.9 * state + 0.2 * (h @ P["B"] + P["c"]))
        if keep:
            records.append((f, h, nxt))
        state = nxt
    return state, records


def nca_loss(P, state, seedmap, target, steps=20):
    final, records = nca_rollout(P, state, seedmap, steps, True)
    error = final[..., 0] - target
    loss = 0.5 * np.mean(error**2)
    grad = np.zeros_like(final)
    grad[..., 0] = error / error.size
    G = {k: np.zeros_like(v) for k, v in P.items()}
    channels = state.shape[-1]
    for f, h, nxt in records[::-1]:
        dz = grad * (1 - nxt**2)
        G["B"] += 0.2 * h.reshape(-1, h.shape[-1]).T @ dz.reshape(-1, channels)
        G["c"] += 0.2 * dz.sum((0, 1))
        dh = 0.2 * (dz @ P["B"].T) * (1 - h**2)
        G["A"] += f.reshape(-1, f.shape[-1]).T @ dh.reshape(-1, h.shape[-1])
        G["b"] += dh.sum((0, 1))
        df = dh @ P["A"].T
        direct, mean, dx, dy = np.split(df[..., :-1], 4, axis=-1)
        mean_back = _nca_filters(mean)[0]
        # Central differences are skew-adjoint on the periodic grid.
        dx_back = -_nca_filters(dx)[1]
        dy_back = -_nca_filters(dy)[2]
        grad = 0.9 * dz + direct + mean_back + dx_back + dy_back
    return float(loss), G


def nca_train(target, steps=550, seed=17):
    """Six-channel local neural rule; BPTT from seed and damaged rollouts."""
    rng = np.random.default_rng(seed)
    channels, hidden = 6, 16
    P = dict(
        A=rng.normal(0, 0.15, (4 * channels + 1, hidden)),
        b=np.zeros(hidden),
        B=rng.normal(0, 0.08, (hidden, channels)),
        c=np.zeros(channels),
    )
    opt = Adam(P, 0.008)
    seedmap = np.zeros_like(target)
    seedmap[target.shape[0] // 2, target.shape[1] // 2] = 1
    initial = np.zeros((*target.shape, channels))
    initial[..., 1] = seedmap
    losses = []
    grown = initial.copy()
    for k in range(steps):
        state = initial.copy()
        # Train damaged matured states too; the target stays outside the rule.
        if k > 100 and k % 3 == 0:
            state = grown.copy()
            row, col = rng.integers(1, 9, 2)
            state[row : row + 4, col : col + 4] = 0
        loss, G = nca_loss(P, state, seedmap, target, 24)
        norm = np.sqrt(sum(np.sum(g * g) for g in G.values()))
        opt.step(P, {key: g * min(1.0, 1.0 / max(norm, 1e-12)) for key, g in G.items()})
        losses.append(loss)
        if k % 10 == 0:
            grown, _ = nca_rollout(P, initial.copy(), seedmap, 32)
    state = initial.copy()
    frames = []
    for t in range(65):
        if t == 33:
            state[3:9, 3:9] = 0
        frames.append(state[..., 0].copy())
        state, _ = nca_rollout(P, state, seedmap, 1)
    return dict(loss=losses, frames=frames, **P)


def cellular():
    def build():
        from .mnist import load

        x, y, _, _ = load()
        i = int(np.flatnonzero(y == 0)[0])
        target = x[i].reshape(14, 2, 14, 2).mean((1, 3))
        d = nca_train(target)
        d["target"] = target
        return d

    d = _cache("neural-cellular", build, steps=550, unroll=24, channels=6, target_digit=0)
    frames = d["frames"]
    mse = ((frames - d["target"]) ** 2).mean((1, 2))
    return _study(
        "A learned cellular update rule",
        "One MNIST training image; six-channel rule fitted through 24 growth steps",
        [
            _images(
                "Grow from a persistent seed",
                "Six cell-state channels read local differences and a persistent central seed.",
                frames[[0, 3, 12, 32]],
                ["seed", "step 3", "step 12", "step 32"],
            ),
            _curve(
                "Train by backpropagation through time",
                "Seeded growth and damaged mature states share one learned local update rule.",
                d["loss"],
                ["training MSE"],
            ),
            _images(
                "Damage and regrow",
                "The same autonomous local update continues after part of the field is erased.",
                frames[[32, 33, 38, 64]],
                ["before", "damaged", "+5 steps", "+31 steps"],
            ),
            _curve(
                "Measure the reconstruction honestly",
                "The full target is shown only to the training loss, never to the local rule.",
                mse,
                ["target MSE"],
                notes=("Final pixel MSE: %.4f" % mse[-1],),
            ),
        ],
        mnist=True,
    )


def ode_field(P, x):
    h = np.tanh(x @ P["W"] + P["b"])
    return h @ P["U"] + P["v"], h


def ode_loss(P, x, target, steps=12):
    dt = 1 / steps
    states = [x]
    hidden = []
    for _ in range(steps):
        v, h = ode_field(P, states[-1])
        hidden.append(h)
        states.append(states[-1] + dt * v)
    grad = (states[-1] - target) / len(x)
    loss = 0.5 * np.mean(np.sum((states[-1] - target) ** 2, 1))
    G = {k: np.zeros_like(v) for k, v in P.items()}
    for i in range(steps - 1, -1, -1):
        z = dt * grad
        h = hidden[i]
        dh = (z @ P["U"].T) * (1 - h * h)
        G["U"] += h.T @ z
        G["v"] += z.sum(0)
        G["W"] += states[i].T @ dh
        G["b"] += dh.sum(0)
        grad = grad + dh @ P["W"].T
    return float(loss), G, np.array(states)


def ode_adjoint(P, final, target, steps):
    """Continuous adjoint integrated backwards with Euler, storing one state.

    Finite step error is explicit: unlike the exact discrete BPTT above, this
    reverse integration approximates the continuous equation's sensitivity.
    """
    dt = 1.0 / steps
    state = final.copy()
    adjoint = (state - target) / len(state)
    G = {k: np.zeros_like(v) for k, v in P.items()}
    for _ in range(steps):
        velocity, h = ode_field(P, state)
        dz = dt * adjoint
        dh = (dz @ P["U"].T) * (1 - h * h)
        G["U"] += h.T @ dz
        G["v"] += dz.sum(0)
        G["W"] += state.T @ dh
        G["b"] += dh.sum(0)
        adjoint += dh @ P["W"].T
        state -= dt * velocity
    return G, state


def neural_ode():
    def build():
        rng = np.random.default_rng(17)
        x = rng.normal(0, 0.45, (96, 2))
        # Paired synthetic warp: a smooth rotation plus radial expansion.
        a = np.arctan2(x[:, 1], x[:, 0]) + 0.8 * np.linalg.norm(x, axis=1)
        r = np.linalg.norm(x, axis=1) * (1 + 0.4 * np.linalg.norm(x, axis=1))
        target = np.c_[r * np.cos(a), r * np.sin(a)]
        P = dict(
            W=rng.normal(0, 0.3, (2, 16)),
            b=np.zeros(16),
            U=rng.normal(0, 0.1, (16, 2)),
            v=np.zeros(2),
        )
        opt = Adam(P, 0.015)
        losses = []
        for _ in range(241):
            loss, G, _ = ode_loss(P, x, target)
            losses.append(loss)
            opt.step(P, G)
        depths = np.array([2, 4, 8, 16, 32, 64])
        errors = []
        finals = []
        adjoint_errors = []
        reverse_errors = []
        for depth in depths:
            loss, exact, states = ode_loss(P, x, target, int(depth))
            approximate, start = ode_adjoint(P, states[-1], target, int(depth))
            adjoint_errors.append(
                np.linalg.norm(_pack(approximate) - _pack(exact))
                / max(np.linalg.norm(_pack(exact)), 1e-12)
            )
            reverse_errors.append(np.linalg.norm(start - x) / np.linalg.norm(x))
            errors.append(loss)
            finals.append(states[-1])
        _, _, states = ode_loss(P, x, target, 32)
        return dict(
            x=x,
            target=target,
            states=states,
            loss=losses,
            depth=depths,
            error=errors,
            memory=depths * len(x) * 2 * 8,
            adjoint_errors=adjoint_errors,
            reverse_errors=reverse_errors,
        )

    d = _cache("neural-ode", build, steps=241, hidden=16, train_depth=12, adjoint="Euler")
    # Reverse integration stores one state and reports its finite-step gradient error.
    return _study(
        "Residual steps approach a neural ODE",
        "A learned autonomous vector field on a paired synthetic two-dimensional warp",
        [
            _points(
                "Learn a continuous flow",
                "Each point follows x ← x + Δt f(x), with one shared learned vector field.",
                d["states"][::4].reshape(-1, 2),
                np.repeat(np.arange(9), len(d["x"])),
            ),
            _curve(
                "Fit the vector field",
                "Backpropagation differentiates through twelve Euler residual steps.",
                d["loss"],
                ["endpoint MSE"],
            ),
            _curve(
                "Reduce the Euler step size",
                "The same weights are integrated with increasing depth.",
                d["error"],
                ["endpoint MSE"],
                d["depth"],
            ),
            _bars(
                "Stored-state memory",
                "Backprop stores states; a continuous adjoint trades storage for recomputation.",
                [d["memory"][-1], len(d["x"]) * 2 * 8],
                ["64 stored steps", "one state"],
                (
                    "Reverse-gradient error at 64 steps: %.3g" % d["adjoint_errors"][-1],
                    "Backward Euler stores one state; error is measured.",
                ),
            ),
        ],
    )


def evolution():
    def build():
        rng = np.random.default_rng(17)
        H = np.array([[4.0, 1.5], [1.5, 1.0]])
        paths = []
        losses = []
        alignment = []
        for method in range(2):
            w = np.array([2.0, -2.0])
            path = []
            ls = []
            cs = []
            for step in range(101):
                true = H @ w
                path.append(w.copy())
                ls.append(0.5 * w @ H @ w)
                if method:
                    eps = rng.normal(size=(64, 2))
                    plus = w + 0.05 * eps
                    minus = w - 0.05 * eps
                    fp = np.einsum("ni,ij,nj->n", plus, H, plus) / 2
                    fm = np.einsum("ni,ij,nj->n", minus, H, minus) / 2
                    estimate = ((fp - fm)[:, None] * eps).mean(0) / 0.1
                else:
                    estimate = true
                cs.append(
                    estimate @ true / max(np.linalg.norm(estimate) * np.linalg.norm(true), 1e-12)
                )
                if step == 100:
                    break
                w -= 0.12 * estimate
            paths.append(path)
            losses.append(ls)
            alignment.append(cs)
        x, y, _, _ = _digit_data(48)
        P = mlp_init(49, 5, 10, rng)
        theta = _pack(P)
        true = _pack(mlp_loss(P, x, y)[1])
        errors = []
        sizes = np.array([8, 32, 128, 512])
        for n in sizes:
            eps = rng.normal(size=(n, len(theta)))
            estimate = np.zeros_like(theta)
            for e in eps:
                a = mlp_loss(_unpack(theta + 0.005 * e, P), x, y, False)[0]
                b = mlp_loss(_unpack(theta - 0.005 * e, P), x, y, False)[0]
                estimate += (a - b) * e / (0.01 * n)
            errors.append(np.linalg.norm(estimate - true) / np.linalg.norm(true))
        network_losses = []
        params = {}
        for method in range(2):
            weights = theta.copy()
            trajectory = []
            for step in range(81):
                model = _unpack(weights, P)
                if method:
                    loss, _ = mlp_loss(model, x, y, False)
                else:
                    loss, grads = mlp_loss(model, x, y)
                trajectory.append(loss)
                if step == 80:
                    break
                if method:
                    perturb = rng.normal(size=(64, len(weights)))
                    gradient = np.zeros_like(weights)
                    for eps in perturb:
                        plus = mlp_loss(_unpack(weights + 0.01 * eps, P), x, y, False)[0]
                        minus = mlp_loss(_unpack(weights - 0.01 * eps, P), x, y, False)[0]
                        gradient += (plus - minus) * eps / (2 * 0.01 * len(perturb))
                else:
                    gradient = _pack(grads)
                weights -= 0.15 * gradient
            network_losses.append(trajectory)
            params.update(
                {"P_method%d_%s" % (method, k): v for k, v in _unpack(weights, P).items()}
            )
        return dict(
            paths=paths,
            loss=losses,
            alignment=alignment,
            sizes=sizes,
            error=errors,
            network_loss=network_losses,
            **params,
        )

    d = _cache(
        "evolution-strategies",
        build,
        population=64,
        sigma=0.05,
        quadratic_updates=100,
        network_updates=80,
        network_sigma=0.01,
    )
    return _study(
        "Learning without backpropagation",
        "Antithetic evolution strategies on an exact quadratic and a small MNIST MLP",
        [
            _points(
                "A population estimates a direction",
                "Paired positive and negative perturbations cancel a constant reward baseline.",
                d["paths"].reshape(-1, 2),
                np.repeat([0, 1], 101),
            ),
            _curve(
                "Optimise the same quadratic",
                "Both methods begin at the same weights and use the same step size.",
                d["loss"],
                ["true gradient", "ES estimate"],
            ),
            _curve(
                "Compare with the exact gradient",
                "Cosine agreement measures whether the population finds a useful direction.",
                d["alignment"],
                ["true gradient", "ES estimate"],
            ),
            _curve(
                "Train a small MNIST network",
                "Each ES update uses 64 antithetic perturbation pairs and no backward pass.",
                d["network_loss"],
                ["true gradient", "ES estimate"],
                notes=("Gradient error at 512 samples: %.2f" % d["error"][-1],),
            ),
        ],
        mnist=True,
    )


FACTORIES = {
    "newton-vs-gradient-descent": newton,
    "edge-of-stability": edge_stability,
    "batch-size-and-noise": batch_noise,
    "weight-averaging": weight_average,
    "second-order-preconditioners": precondition,
    "loss-spikes": loss_spikes,
    "gpt2-retrieval-augmented-generation": retrieval,
    "vector-search-hnsw": hnsw,
    "infonce-temperature": temperature,
    "hard-negatives": hard_negatives,
    "dimensionality-of-embeddings": embedding_width,
    "data-parallel-allreduce": allreduce,
    "tensor-and-pipeline-parallel": tensor_pipeline,
    "zero-sharding": zero_sharding,
    "activation-checkpointing": checkpointing,
    "roofline": roofline,
    "multi-armed-bandits": bandits,
    "policy-gradient": policy_gradient,
    "actor-critic": actor_critic,
    "dqn-replay-buffer": dqn,
    "mcts-tic-tac-toe": mcts,
    "reward-shaping": reward_shaping,
    "hopfield-network": hopfield,
    "neural-cellular-automata": cellular,
    "neural-ode": neural_ode,
    "evolution-strategies": evolution,
}


def make(name):
    """Build one of the measured studies; no publishing or networking beyond model cache."""
    if name == "retrieval-augmented-generation":
        name = "gpt2-" + name
    from .next_systems_real import BUILDERS

    if name in BUILDERS:
        return BUILDERS[name]()
    _RUN_EVIDENCE.clear()
    study = FACTORIES[name]()
    if _RUN_EVIDENCE:
        study.metadata["training_provenance"] = list(_RUN_EVIDENCE)
        study.metadata["evidence_kind"] = "actual NumPy training on MNIST"
    else:
        study.metadata["evidence_kind"] = (
            "direct calculation, search, or measured model evaluation; no learning claim"
        )
    return study

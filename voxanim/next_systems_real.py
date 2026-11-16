"""Dataset-grounded replacements for the TODO systems and RL learning studies.

The reinforcement-learning examples use classification decisions over actual
MNIST observations. They are explicitly narrow decision tasks, not evidence
about navigation or game-playing. Numerical diagrams remain separate examples.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np

from . import next_systems as S
from . import realdata as R
from .cache import cached_npz
from .optim import Adam


@lru_cache(maxsize=1)
def digits():
    x, y, train = R.mnist_split(1024, "train", seed=17)
    xt, yt, test = R.mnist_split(384, "test", seed=73)
    return x, y, xt, yt, dict(train=train, test=test)


def cached(name, build, **settings):
    return {
        k: np.asarray(v)
        for k, v in cached_npz(
            "next-real-" + name,
            dict(
                version=1,
                source=R.source_hash(
                    "next_systems_real.py", "next_systems.py", "realdata.py", "optim.py"
                ),
                seed=17,
                **settings,
            ),
            build,
        ).items()
    }


def finish(name, title, subtitle, panels, d, settings, dataset=None):
    params = {k[2:]: v for k, v in d.items() if k.startswith("P_")}
    history = {k: v for k, v in d.items() if not k.startswith("P_")}
    evidence = R.save_evidence(
        name,
        params,
        history,
        dataset or digits()[4],
        settings,
        ("next_systems_real.py", "next_systems.py", "optim.py"),
    )
    return S._study(
        title,
        subtitle,
        panels,
        training_provenance=evidence,
        evidence_kind="actual NumPy updates on collected data",
        limitations=["A small, seeded dataset experiment; outcomes are not guaranteed."],
    )


def _linear_loss(w, x, y):
    p = S.softmax(x @ w)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()
    p[np.arange(len(y)), y] -= 1
    return float(loss), x.T @ p / len(y)


def precondition(newton=False):
    """A true least-squares fit of one-hot MNIST targets, with known Hessian."""

    def build():
        x, y, xt, yt, _ = digits()
        a, at = np.c_[x, np.ones(len(x))], np.c_[xt, np.ones(len(xt))]
        target = np.eye(10)[y]
        h = a.T @ a / len(a) + 0.02 * np.eye(a.shape[1])
        diagonal_scale = 1 / np.sqrt(np.diag(h))
        diagonal_hessian = diagonal_scale[:, None] * h * diagonal_scale[None, :]
        diagonal_rate = 0.9 / np.linalg.eigvalsh(diagonal_hessian).max()
        rates = (0.9 / np.linalg.eigvalsh(h).max(), 0.08, 1.0)
        losses, tests, weights = [], [], []
        for method in range(3):
            w = np.zeros((a.shape[1], 10))
            parameters = {"w": w}
            opt = Adam(parameters, rates[1])
            ls, ev = [], []
            for step in range(81):
                error = a @ w - target
                ls.append(float(np.mean(np.sum(error * error, axis=1)) / 2 + 0.01 * np.sum(w * w)))
                ev.append(float(np.mean((at @ w).argmax(1) == yt)))
                if step == 80:
                    break
                g = a.T @ error / len(a) + 0.02 * w
                if method == 0:
                    w -= rates[0] * g
                elif method == 1:
                    if newton:
                        opt.step(parameters, {"w": g})
                        w = parameters["w"]
                    else:
                        w -= diagonal_rate * g / np.diag(h)[:, None]
                else:
                    w -= np.linalg.solve(h, g)
            losses.append(ls)
            tests.append(ev)
            weights.append(w)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        conditioning = [
            np.linalg.cond(h),
            np.linalg.cond(diagonal_hessian),
            np.linalg.cond(inverse_root @ h @ inverse_root),
        ]
        return dict(h=h, loss=losses, accuracy=tests, conditioning=conditioning, P_weights=weights)

    name = "newton-vs-gradient-descent" if newton else "second-order-preconditioners"
    d = cached(
        name,
        build,
        updates=80,
        ridge=0.02,
        comparison="newton-steps-v3" if newton else "diagonal-curvature-v3",
    )
    names = ["gradient descent", "Adam" if newton else "diagonal scaling", "full Hessian solve"]
    return finish(
        name,
        "Newton steps versus gradient steps" if newton else "Curvature preconditioning on MNIST",
        "MNIST ridge-regression classifier; full NumPy optimization, 1,024 train / 384 test",
        [
            S._matrix(
                "The data determine curvature",
                "H = XᵀX/N + 0.02 I for this stated quadratic objective.",
                d["h"],
            ),
            S._curve(
                "Actual training objective",
                "Every point is evaluated on the current model before its update.",
                d["loss"],
                names,
            ),
            S._curve(
                "Held-out classification",
                "A lower fitting objective need not give higher digit accuracy.",
                d["accuracy"],
                names,
            )
            if newton
            else S._bars(
                "Conditioning after coordinate scaling",
                "Largest / smallest curvature; diagonal scaling leaves some feature coupling.",
                d["conditioning"],
                ["original", "diagonal", "full whitening"],
            ),
            S._bars(
                "Final test accuracy",
                "The exact Hessian solves this quadratic, not every neural-network objective.",
                d["accuracy"][:, -1],
                names,
            ),
        ],
        d,
        dict(
            updates=80,
            objective="ridge squared error to digit one-hot targets",
            ridge=0.02,
            methods=names,
            diagonal_rate_rule="0.9 / largest eigenvalue of D^-1/2 H D^-1/2",
        ),
    )


def loss_spikes():
    from . import tinylm as T

    def build():
        seq = R.text_sequences(64, 13, "train", 17)
        # Two observed batches from disjoint positions of the same training partition.
        batches = [(seq[:32, :-1], seq[:32, 1:]), (seq[32:, :-1], seq[32:, 1:])]
        cfg = T.Config(vocab=len(R.vocabulary()), width=16, heads=2, kv_heads=2, ff=24)
        losses, norms, moments, params = [], [], [], {}
        for j, (beta2, eps, clip) in enumerate(
            ((0.999, 1e-8, None), (0.9, 1e-8, None), (0.999, 1e-3, 1.0))
        ):
            p = T.init(cfg, 17)
            m = {k: np.zeros_like(v) for k, v in p.items()}
            v = {k: q.copy() for k, q in m.items()}
            ls, ns, vs = [], [], []
            for i in range(141):
                x, y = batches[int(i >= 60)]
                loss, g = T.loss_grads(p, x, y, cfg)
                norm = np.sqrt(sum(float((z * z).sum()) for z in g.values()))
                ls.append(loss)
                ns.append(norm)
                vs.append(np.sqrt(sum(float(q.sum()) for q in v.values())))
                factor = min(1, clip / max(norm, 1e-12)) if clip else 1
                for k in p:
                    z = g[k] * factor
                    m[k] = 0.9 * m[k] + 0.1 * z
                    v[k] = beta2 * v[k] + (1 - beta2) * z * z
                    p[k] -= (
                        0.025
                        * (m[k] / (1 - 0.9 ** (i + 1)))
                        / (np.sqrt(v[k] / (1 - beta2 ** (i + 1))) + eps)
                    )
            losses.append(ls)
            norms.append(ns)
            moments.append(vs)
            params.update({"P_%d_%s" % (j, k): z for k, z in p.items()})
        return dict(loss=losses, norm=norms, moment=moments, **params)

    d = cached("loss-spikes", build, updates=141, switch=60, dataset=R.TEXT_SHA256)
    names = ["β₂=.999", "β₂=.9", "clip + ε=.001"]
    return finish(
        "loss-spikes",
        "Response to a changed text batch",
        "Tiny Shakespeare; real next-character targets; batch switches at update 60",
        [
            S._curve(
                "Actual training losses",
                "Only the observed text batch changes; no target labels are fabricated.",
                d["loss"],
                names,
            ),
            S._curve(
                "Measured gradient norms",
                "The same two text batches are used by all optimizer configurations.",
                d["norm"],
                names,
            ),
            S._curve(
                "Second-moment state",
                "The optimizer state is recorded at each real update.",
                d["moment"],
                names,
            ),
            S._bars(
                "Last ten updates",
                "This controlled batch switch is not a spontaneous instability claim.",
                d["loss"][:, -10:].mean(1),
                names,
            ),
        ],
        d,
        dict(updates=141, switch=60, learning_rate=0.025),
        R.text_metadata(),
    )


def embeddings():
    def build():
        x, y, xt, yt, _ = digits()
        dims = np.array([2, 4, 8, 16, 32, 64, 128, 256])
        scores = []
        traces = []
        params = {}
        nested = []
        for dim in dims:
            p = {"W": np.random.default_rng(17).normal(0, 0.1, (49, int(dim)))}
            # Separate sampling from shape-dependent parameter initialization.
            rng = np.random.default_rng(73)
            opt = Adam(p, 0.008)
            ls = []
            for _ in range(241):
                ids = rng.choice(len(x), 32, False)
                a = x[ids] + rng.normal(0, 0.03, (32, 49))
                b = x[ids] + rng.normal(0, 0.03, (32, 49))
                loss, g, _, _ = S.infonce_loss(p["W"], a, b, 0.2)
                opt.step(p, {"W": g})
                ls.append(loss)

            def accuracy(w):
                z = x @ w
                z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
                q = xt @ w
                q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-10)
                return np.mean(y[(q @ z.T).argmax(1)] == yt)

            scores.append(accuracy(p["W"]))
            traces.append(ls)
            params["P_width%d" % dim] = p["W"].copy()
            if dim == 256:
                for _ in range(121):
                    ids = rng.choice(len(x), 32, False)
                    a = x[ids] + rng.normal(0, 0.03, (32, 49))
                    b = x[ids] + rng.normal(0, 0.03, (32, 49))
                    grad = np.zeros_like(p["W"])
                    for width in (4, 16, 64, 256):
                        _, g, _, _ = S.infonce_loss(p["W"][:, :width], a, b, 0.2)
                        grad[:, :width] += g / 4
                    opt.step(p, {"W": grad})
                nested = [accuracy(p["W"][:, :width]) for width in dims]
                params["P_nested"] = p["W"]
        return dict(dims=dims, accuracy=scores, nested=nested, loss=traces, **params)

    d = cached("embedding-width", build, updates=241, nested_updates=121)
    return finish(
        "embedding-width",
        "How wide should an image embedding be?",
        "MNIST paired-view contrastive training; official held-out images",
        [
            S._curve(
                "Actual contrastive training",
                "Each dimension trains on the same measured images and sampled noise.",
                d["loss"][[0, 3, 7]],
                ["2 coordinates", "16 coordinates", "256 coordinates"],
            ),
            S._curve(
                "Test nearest-neighbour accuracy",
                "No generated semantic categories are used to score the representation.",
                d["accuracy"],
                ["separate models"],
                d["dims"],
            ),
            S._curve(
                "Nested prefix training",
                "The nested model receives 121 additional updates; this extra budget is explicit.",
                [d["accuracy"], d["nested"]],
                ["separate", "nested prefixes"],
                d["dims"],
            ),
            S._bars(
                "Float32 storage per vector",
                "This panel is exact byte arithmetic, not measured device memory.",
                d["dims"] * 4,
                [str(n) for n in d["dims"]],
            ),
        ],
        d,
        dict(updates=241, nested_updates=121),
    )


def neural_ode():
    def build():
        x, _, xt, _, _ = digits()
        mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        scale = np.std((x - mean) @ vt[:2].T, axis=0)
        target = ((x - mean) @ vt[:2].T) / scale
        target_test = ((xt - mean) @ vt[:2].T) / scale
        rng = np.random.default_rng(17)
        noisy = target + rng.normal(0, 0.3, target.shape)
        observed = target_test + rng.normal(0, 0.3, target_test.shape)
        p = dict(
            W=rng.normal(0, 0.2, (2, 16)),
            b=np.zeros(16),
            U=rng.normal(0, 0.1, (16, 2)),
            v=np.zeros(2),
        )
        opt = Adam(p, 0.008)
        losses = []
        for _ in range(241):
            ids = rng.choice(len(x), 64, False)
            loss, g, _ = S.ode_loss(p, noisy[ids], target[ids])
            losses.append(loss)
            opt.step(p, g)
        depths = np.array([2, 4, 8, 16, 32, 64])
        errors = []
        numerical = []
        adj = []
        reference = S.ode_loss(p, observed, target_test, 256)[2][-1]
        for n in depths:
            loss, g, states = S.ode_loss(p, observed, target_test, int(n))
            ga, _ = S.ode_adjoint(p, states[-1], target_test, int(n))
            errors.append(loss)
            numerical.append(np.mean((states[-1] - reference) ** 2))
            adj.append(
                np.linalg.norm(S._pack(g) - S._pack(ga)) / max(np.linalg.norm(S._pack(g)), 1e-12)
            )
        states = S.ode_loss(p, observed[:24], target_test[:24], 32)[2]
        return dict(
            loss=losses,
            depths=depths,
            error=errors,
            numerical=numerical,
            adjoint=adj,
            states=states,
            P_pca=vt[:2],
            P_mean=mean,
            P_scale=scale,
            **{"P_" + k: v for k, v in p.items()},
        )

    d = cached("neural-ode", build, updates=241, pca_fit="train only", corruption_sd=0.3)
    return finish(
        "neural-ode",
        "A neural ODE fitted to measured-image coordinates",
        "MNIST training-only PCA; learned denoising, not a physical time-evolution claim",
        [
            S._points(
                "Held-out denoising paths",
                "States come from numerical integration of the same learned vector field.",
                d["states"][::4].reshape(-1, 2),
                np.repeat(np.arange(9), 24),
            ),
            S._curve(
                "Actual NumPy optimization",
                "A minibatch endpoint loss is evaluated before every parameter update.",
                d["loss"],
                ["training endpoint loss"],
            ),
            S._curve(
                "Model error and numerical error",
                "Endpoint error and error against a 256-step reference are distinct measurements.",
                [d["error"], d["numerical"]],
                ["held-out endpoint", "integration reference"],
                d["depths"],
            ),
            S._curve(
                "Adjoint gradient discrepancy",
                "Explicit Euler in reverse time approximates the continuous adjoint.",
                d["adjoint"],
                ["relative gradient error"],
                d["depths"],
            ),
        ],
        d,
        dict(updates=241, training_depth=12, pca_fit="training only"),
    )


def retrieval():
    from . import gpt2, gpt2_hooks

    tokenizer = gpt2.Tokenizer()
    lines = [line.strip() for line in R.text("test").splitlines() if 40 < len(line.strip()) < 130]
    passages = lines[:5]
    query = " ".join(passages[0].split()[:5])

    def build():
        train_lines = [
            re.findall(r"[a-z]+", s.lower()) for s in R.text("train").splitlines() if s.strip()
        ]
        counts = Counter(w for line in train_lines for w in set(line))
        words = [w for w, _ in counts.most_common(2000)]
        index = {w: i for i, w in enumerate(words)}
        idf = np.array([np.log((len(train_lines) + 1) / (counts[w] + 1)) + 1 for w in words])
        vectors = []
        for line in passages + [query]:
            v = np.zeros(len(words))
            for word in re.findall(r"[a-z]+", line.lower()):
                if word in index:
                    v[index[word]] += 1
            v *= idf
            v /= max(np.linalg.norm(v), 1e-12)
            vectors.append(v)
        vectors = np.array(vectors)
        scores = vectors[:-1] @ vectors[-1]
        order = np.argsort(-scores)
        prompt_ids = tokenizer.encode(query)
        full_ids = tokenizer.encode(passages[0])
        answer = full_ids[len(prompt_ids)]
        prompts = [
            query,
            passages[int(order[0])] + "\n" + query,
            passages[int(order[-1])] + "\n" + query,
        ]
        predictions = [
            gpt2.softmax(gpt2_hooks.forward(tokenizer.encode(p))["logits"][-1]) for p in prompts
        ]
        candidates = np.unique(np.r_[answer, np.argsort(predictions[0])[-7:]])
        return dict(
            scores=scores,
            similarity=vectors @ vectors.T,
            probs=[p[answer] for p in predictions],
            tops=np.array(predictions)[:, candidates],
            candidates=candidates,
            answer=answer,
            P_idf=idf,
        )

    d = cached("retrieval", build, dataset=R.TEXT_SHA256, passages=passages, query=query)
    result = finish(
        "retrieval",
        "Retrieval from real text",
        "Tiny Shakespeare passages; training-text TF–IDF; actual pretrained GPT-2 probabilities",
        [
            S._matrix(
                "A prefix queries five passages",
                "The query contains only the observed prefix; its continuation is withheld.",
                d["similarity"],
                (query,),
            ),
            S._bars(
                "Retrieve by text similarity",
                "IDF statistics are fitted on the separate training-text partition.",
                d["scores"],
                ["passage %d" % (i + 1) for i in range(5)],
            ),
            S._bars(
                "Observed continuation token",
                "Relevant and low-ranked context alter an actual GPT-2 next-token probability.",
                d["probs"],
                ["prefix only", "top retrieval", "last retrieval"],
            ),
            S._matrix(
                "Same token columns in every condition",
                "The candidate IDs are fixed across the three measured distributions.",
                d["tops"],
                tuple(str(int(i)) for i in d["candidates"][:3]),
            ),
        ],
        d,
        dict(
            method="TF–IDF counts, no gradient training; pretrained GPT-2 evaluation",
            query=query,
            passages=passages,
        ),
        dict(train=R.text_metadata(), test=R.text_metadata("test")),
    )
    result.metadata["evidence_kind"] = "real-text fitted statistics and pretrained evaluation"
    return result


def hnsw():
    def build():
        x, _y, xt, _yt, _ = digits()
        rng = np.random.default_rng(17)
        p = {"W": rng.normal(0, 0.1, (49, 12))}
        opt = Adam(p, 0.008)
        losses = []
        for _ in range(161):
            ids = rng.choice(len(x), 32, False)
            a = x[ids] + rng.normal(0, 0.03, (32, 49))
            b = x[ids] + rng.normal(0, 0.03, (32, 49))
            loss, g, _, _ = S.infonce_loss(p["W"], a, b, 0.2)
            opt.step(p, {"W": g})
            losses.append(loss)
        u = x[:256] @ p["W"]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        q = xt[:64] @ p["W"]
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        graph = S.HNSW(u)
        exact = ((q[:, None] - u[None]) ** 2).sum(-1).argmin(1)
        efs = np.array([1, 2, 4, 8, 16, 32])
        recall = []
        work = []
        for ef in efs:
            runs = [graph.query(v, int(ef)) for v in q]
            recall.append(np.mean([r[0] == e for r, e in zip(runs, exact)]))
            work.append(np.mean([r[1] for r in runs]))
        _, _, vt = np.linalg.svd(u - u.mean(0), full_matrices=False)
        xy = (u - u.mean(0)) @ vt[:2].T
        path = graph.query(q[0], 8)[2]
        return dict(
            xy=xy,
            levels=graph.levels,
            path=xy[path],
            recall=recall,
            work=work,
            efs=efs,
            loss=losses,
            P_encoder=p["W"],
        )

    d = cached("hnsw", build, encoder_updates=161, index=256, queries=64)
    return finish(
        "hnsw",
        "Hierarchical search over learned image vectors",
        "MNIST-trained contrastive encoder; held-out query images; HNSW-style search",
        [
            S._points(
                "The indexed training images",
                "The plot is a projection; distances use all twelve learned coordinates.",
                d["xy"],
                d["levels"],
            ),
            S._curve(
                "Train the encoder",
                "Paired noisy views come from actual MNIST observations.",
                d["loss"],
                ["contrastive loss"],
            ),
            S._points(
                "One search path",
                "The displayed path is the actual traversal for one held-out query.",
                d["path"],
            ),
            S._curve(
                "Recall and measured search work",
                "Compare exact neighbors; count distance calls relative to index size.",
                [d["recall"], d["work"] / 256],
                ["exact-neighbor recall", "fraction examined"],
                d["efs"],
            ),
        ],
        d,
        dict(encoder_updates=161, index=256, queries=64),
    )


def decision_run(mode="reinforce", baseline=True, bonus=0.0, updates=500):
    """Two observed views of each digit; actions predict its measured label.

    The first state hides half the image, the second exposes the whole image.
    Transition to the second view is action-independent. Rewards are label
    correctness, optionally plus an explicit abstention incentive.
    """
    x, y, xt, yt, _ = digits()
    rng = np.random.default_rng(17)
    a = np.c_[x, np.ones(len(x))]
    at = np.c_[xt, np.ones(len(xt))]
    actions = 11 if bonus else 10
    p = {"W": rng.normal(0, 0.01, (50, actions)), "v": np.zeros(50)}
    opt = Adam(p, 0.015)
    accuracy = []
    reward_trace = []
    variance = []
    td_trace = []
    for step in range(updates):
        ids = rng.choice(len(x), 64, False)
        full = a[ids]
        partial = full.copy()
        partial[:, :24] = 0
        states = (partial, full)
        probs = [S.softmax(z @ p["W"]) for z in states]
        acts = [np.array([rng.choice(actions, p=q) for q in ps]) for ps in probs]
        rewards = [(act == y[ids]).astype(float) + bonus * (act == 10) for act in acts]
        values = [z @ p["v"] for z in states]
        gW = np.zeros_like(p["W"])
        gv = np.zeros_like(p["v"])
        estimates = []
        for i in (0, 1):
            if mode == "actor":
                target = rewards[i] + (0.9 * values[1] if i == 0 else 0)
            else:
                target = rewards[i] + (0.9 * rewards[1] if i == 0 else 0)
            advantage = target - values[i] if baseline else target
            direction = np.eye(actions)[acts[i]] - probs[i]
            # Discount both the return and the time at which its score occurs.
            weighted_score = (0.9**i) * advantage[:, None] * direction
            gW -= states[i].T @ weighted_score / len(ids)
            gv += states[i].T @ (values[i] - target) / len(ids)
            estimates.append(weighted_score)
            if i == 0:
                td_trace.append(float(np.mean(advantage)))
        opt.step(p, dict(W=gW / 2, v=gv / 2))
        reward_trace.append(float(np.mean(rewards)))
        # This is explicitly sampled update-component dispersion, not an
        # estimator-variance theorem or a controlled baseline-variance test.
        variance.append(float(np.var(np.concatenate(estimates), axis=0).mean()))
        if step % 10 == 0 or step == updates - 1:
            accuracy.append(float(np.mean((at @ p["W"]).argmax(1) == yt)))
    return dict(
        reward=reward_trace,
        dispersion=variance,
        accuracy=accuracy,
        advantage=td_trace,
        policy=S.softmax(at[:12] @ p["W"]),
        P_W=p["W"],
        P_value=p["v"],
        labels=yt[:12],
    )


def policy_gradient(actor=False):
    name = "actor-critic" if actor else "policy-gradient"

    def build():
        runs = [
            decision_run("actor" if actor else "reinforce", b)
            for b in ((True,) if actor else (False, True))
        ]
        out = {
            k: np.array([r[k] for r in runs])
            for k in ("reward", "dispersion", "accuracy", "advantage")
        }
        out["policy"] = runs[-1]["policy"]
        for j, r in enumerate(runs):
            out.update({"P_%d_%s" % (j, k): v for k, v in r.items() if k.startswith("P_")})
        return out

    d = cached(name, build, updates=500, two_views=True)
    names = ["actor + critic"] if actor else ["no baseline", "learned baseline"]
    return finish(
        name,
        "An actor and critic on measured digits"
        if actor
        else "Policy gradients from measured digit rewards",
        "MNIST two-view decision task; sampled label rewards; no gridworld learning claim",
        [
            S._matrix(
                "Learned action probabilities",
                "Rows are held-out images; columns are digit predictions 0–9.",
                d["policy"],
            ),
            S._curve(
                "Observed training rewards",
                "The policy samples actions; feedback is correctness against actual MNIST labels.",
                d["reward"],
                names,
            ),
            S._curve(
                "Held-out full-image accuracy",
                "Greedy evaluation after every ten updates; the test set never updates weights.",
                d["accuracy"],
                names,
            ),
            S._curve(
                "TD advantage" if actor else "Sampled update dispersion",
                "The critic bootstraps from the second view; terminal value is zero."
                if actor
                else "Batch dispersion is descriptive, not a controlled variance comparison.",
                d["advantage"] if actor else d["dispersion"],
                names,
            ),
        ],
        d,
        dict(
            updates=500,
            batch=64,
            discount=0.9,
            views="half image then full image",
            reward="correct measured label",
        ),
    )


def reward_shaping():
    def build():
        runs = [decision_run("actor", True, b) for b in (0.0, 1.2)]
        out = {k: np.array([r[k] for r in runs]) for k in ("reward", "accuracy")}
        out["policy"] = runs[-1]["policy"]
        for j, r in enumerate(runs):
            out.update({"P_%d_%s" % (j, k): v for k, v in r.items() if k.startswith("P_")})
        return out

    d = cached("reward-shaping", build, updates=500, abstention_bonus=1.2)
    return finish(
        "reward-shaping",
        "A reward bonus can change the task",
        "Real MNIST label rewards; explicit +1.2 abstention incentive in the second run",
        [
            S._curve(
                "Reward optimized during training",
                "One policy earns correctness reward; the other is also paid to abstain.",
                d["reward"],
                ["correctness only", "abstention bonus"],
            ),
            S._curve(
                "Evaluate the original objective",
                "Test accuracy counts abstention as failing to identify the digit.",
                d["accuracy"],
                ["correctness only", "abstention bonus"],
            ),
            S._matrix(
                "Actions on held-out images",
                "The last column is abstention; its bonus is configured, not measured utility.",
                d["policy"],
            ),
            S._bars(
                "Final held-out accuracy",
                "This measured bonus changes the task; it is not potential-based shaping.",
                d["accuracy"][:, -1],
                ["correctness only", "abstention bonus"],
            ),
        ],
        d,
        dict(updates=500, reward="MNIST correctness", configured_bonus=1.2),
    )


def dqn():
    def build():
        x, y, xt, yt, _ = digits()
        a = np.c_[x, np.ones(len(x))]
        at = np.c_[xt, np.ones(len(xt))]
        # Replay stores two-view transitions from collected images, not a
        # simulated spatial navigation process.
        partial = a.copy()
        partial[:, :24] = 0
        states = np.concatenate([partial, a])
        nexts = np.concatenate([a, np.zeros_like(a)])
        labels = np.tile(y, 2)
        done = np.repeat([False, True], len(a))
        results = []
        losses = []
        params = {}
        for run, (replay, target_net) in enumerate(
            ((False, False), (True, False), (False, True), (True, True))
        ):
            rng = np.random.default_rng(17)
            p = {"W": rng.normal(0, 0.01, (50, 10))}
            target = p["W"].copy()
            opt = Adam(p, 0.008)
            ls = []
            acc = []
            order = rng.permutation(len(states))
            pool = []
            for step in range(500):
                fresh = order[(np.arange(64) + step * 64) % len(order)]
                actions = (states[fresh] @ p["W"]).argmax(1)
                explore = rng.random(64) < 0.25
                actions[explore] = rng.integers(10, size=explore.sum())
                outcomes = (actions == labels[fresh]).astype(float)
                new_transitions = list(zip(fresh.tolist(), actions.tolist(), outcomes.tolist()))
                pool.extend(new_transitions)
                pool = pool[-2048:]
                chosen = (
                    [pool[i] for i in rng.integers(len(pool), size=64)]
                    if replay
                    else new_transitions
                )
                ids = np.array([row[0] for row in chosen], int)
                acts = np.array([row[1] for row in chosen], int)
                reward = np.array([row[2] for row in chosen])
                q = states[ids] @ p["W"]
                bootstrap = (nexts[ids] @ (target if target_net else p["W"])).max(1)
                targets = reward + 0.9 * (~done[ids]) * bootstrap
                err = q[np.arange(64), acts] - targets
                dq = np.zeros_like(q)
                dq[np.arange(64), acts] = err / 64
                opt.step(p, {"W": states[ids].T @ dq})
                ls.append(float(np.mean(err**2) / 2))
                if (step + 1) % 25 == 0:
                    target = p["W"].copy()
                if step % 10 == 0 or step == 499:
                    acc.append(float(np.mean((at @ p["W"]).argmax(1) == yt)))
            results.append(acc)
            losses.append(ls)
            params["P_run%d" % run] = p["W"]
        return dict(accuracy=results, loss=losses, final_replay=np.array(pool), **params)

    d = cached("dqn", build, updates=500, batch=64, target_period=25)
    labels = ["online", "replay", "target", "replay + target"]
    return finish(
        "dqn",
        "Replay and target networks on observed digits",
        "Linear Q-learning on MNIST two-view transitions; 500 matched updates",
        [
            S._curve(
                "Measured TD loss",
                "Each transition records image, action, label reward, next view and termination.",
                d["loss"],
                labels,
            ),
            S._curve(
                "Held-out classification accuracy",
                "Two-by-two ablation separates replay from delayed target weights.",
                d["accuracy"],
                labels,
            ),
            S._bars(
                "Final held-out performance",
                "A stabilization method is not assumed to win this small task.",
                d["accuracy"][:, -1],
                labels,
            ),
            S._matrix(
                "The learned action-value weights",
                "These weights came from sampled temporal-difference updates on real observations.",
                d["P_run3"],
            ),
        ],
        d,
        dict(
            updates=500,
            batch=64,
            target_period=25,
            discount=0.9,
            model="linear action-value function, 50 inputs and 10 actions",
            reward="correct MNIST digit",
        ),
    )


def bandits():
    def build():
        x, y, xt, yt, _ = digits()
        xv, yv, _ = R.mnist_split(1000, "validation", seed=19)
        rng = np.random.default_rng(17)
        reward_table = []
        params = {}
        accuracy = []
        for j, count in enumerate((1, 2, 4, 7, 10, 15, 21, 28, 40, 49)):
            selected = np.argsort(x.var(0))[-count:]
            a = np.c_[x[:, selected], np.ones(len(x))]
            av = np.c_[xv[:, selected], np.ones(len(xv))]
            at = np.c_[xt[:, selected], np.ones(len(xt))]
            w = np.zeros((count + 1, 10))
            parameters = {"w": w}
            opt = Adam(parameters, 0.03)
            for _ in range(100):
                ids = rng.choice(len(a), 64, False)
                _, g = _linear_loss(w, a[ids], y[ids])
                opt.step(parameters, {"w": g})
                w = parameters["w"]
            reward_table.append((av @ w).argmax(1) == yv)
            accuracy.append(np.mean((at @ w).argmax(1) == yt))
            params["P_arm%d" % j] = w
            params["feature_indices%d" % j] = selected
        table = np.array(reward_table)
        means = table.mean(1)
        regrets = []
        counts = []
        for method in range(3):
            traces = []
            ns = []
            for seed in range(8):
                r = np.random.default_rng(seed + 17)
                n = np.zeros(10)
                wins = np.zeros(10)
                regret = []
                total = 0.0
                for t in range(1000):
                    q = wins / np.maximum(n, 1)
                    if method == 0:
                        arm = int(r.integers(10)) if r.random() < 0.1 else int(q.argmax())
                    elif method == 1:
                        arm = (
                            int(np.flatnonzero(n == 0)[0])
                            if np.any(n == 0)
                            else int(np.argmax(q + np.sqrt(2 * np.log(t + 2) / n)))
                        )
                    else:
                        arm = int(np.argmax(r.beta(wins + 1, n - wins + 1)))
                    outcome = table[arm, int(r.integers(table.shape[1]))]
                    n[arm] += 1
                    wins[arm] += outcome
                    total += means.max() - means[arm]
                    regret.append(total)
                traces.append(regret)
                ns.append(n)
            regrets.append(np.mean(traces, 0))
            counts.append(np.mean(ns, 0))
        return dict(
            means=means, test=accuracy, regret=regrets, counts=counts, reward_table=table, **params
        )

    d = cached(
        "bandits",
        build,
        arms=10,
        classifier_updates=100,
        pulls=1000,
        seeds=8,
        training="persistent-adam-v2",
    )
    return finish(
        "bandits",
        "Choose among ten trained digit classifiers",
        "MNIST classifiers; bandit feedback resamples their observed validation successes",
        [
            S._bars(
                "Measured arm success rates",
                "Each rate is observed validation accuracy, not a prescribed success rate.",
                d["means"],
                [str(i) for i in range(10)],
            ),
            S._curve(
                "Regret against empirical arm means",
                "Feedback uses recorded label outcomes; average eight action-selection seeds.",
                d["regret"],
                ["ε-greedy", "UCB", "Thompson"],
            ),
            S._matrix(
                "Observed selection counts",
                "Rows are policies and columns identify the trained classifiers.",
                d["counts"],
            ),
            S._bars(
                "Separate test accuracies",
                "These test outcomes are not used as bandit feedback.",
                d["test"],
                [str(i) for i in range(10)],
            ),
        ],
        d,
        dict(classifier_updates=100, pulls=1000, seeds=8, feedback="validation-label correctness"),
        dict(**digits()[4], validation=R.mnist_split(1000, "validation", seed=19)[2]),
    )


BUILDERS = {
    "newton-vs-gradient-descent": lambda: precondition(True),
    "second-order-preconditioners": precondition,
    "loss-spikes": loss_spikes,
    "dimensionality-of-embeddings": embeddings,
    "neural-ode": neural_ode,
    "policy-gradient": policy_gradient,
    "actor-critic": lambda: policy_gradient(True),
    "dqn-replay-buffer": dqn,
    "reward-shaping": reward_shaping,
    "multi-armed-bandits": bandits,
    "gpt2-retrieval-augmented-generation": retrieval,
    "vector-search-hnsw": hnsw,
}

"""The attention and transformer-block experiments proposed in TODO items 1–20."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import cache

import numpy as np

from . import realdata as R
from . import tinylm as T
from .cache import cached_npz
from .study import Panel, Study


def curve(title, caption, ys, labels, notes=(), x=None):
    ys = np.atleast_2d(ys)
    return Panel(
        title,
        caption,
        "curve",
        {"x": np.arange(ys.shape[1]) if x is None else x, "ys": ys, "labels": labels},
        tuple(notes),
    )


def bars(title, caption, values, labels, notes=()):
    return Panel(
        title, caption, "bars", {"values": np.asarray(values), "labels": labels}, tuple(notes)
    )


def matrix(title, caption, values, notes=()):
    return Panel(title, caption, "matrix", {"values": np.asarray(values)}, tuple(notes))


def _task(kind="markov", n=192, length=12, seed=37):
    return text_task(kind, n, length, seed)


def text_task(kind="markov", n=192, length=12, seed=37, split="train"):
    """Real text windows; masks change observations, never fabricate target characters.

    ``markov`` is retained as a private caller-compatible name for next-character
    prediction; the data are now unmodified Tiny Shakespeare text.
    """
    if kind == "markov":
        return R.language_data(n=n, length=length, split=split, seed=seed)
    y = R.text_sequences(n=n, length=length, split=split, seed=seed)
    x, w = y.copy(), np.zeros_like(y, float)
    if kind == "fill":
        x[:, length // 2] = 0
        w[:, length // 2] = 1
    elif kind == "copy":
        x[:, length // 2 :] = 0
        w[:, length // 2 :] = 1
    else:
        raise ValueError(kind)
    return x, y, w


def task_provenance(kind="markov", train_seed=37, test_seed=73):
    return {
        "dataset": R.text_metadata("train"),
        "evaluation_dataset": R.text_metadata("test"),
        "task": {
            "markov": "next-character prediction",
            "fill": "recover one masked character",
            "copy": "complete a fully masked text suffix",
            "denoise": "restore masked text",
        }[kind],
        "source_sha256": R.source_hash("next_attention.py", "realdata.py"),
        "selection": "fixed settings; no validation-based model selection",
        "train_window_seed": train_seed,
        "test_window_seed": test_seed,
        "window_sampling": "without replacement within each source partition",
    }


def run_metadata(runs, task="markov", **extra):
    dataset = runs[0]["provenance"]["settings"]["provenance"] if runs else task_provenance(task)
    return {**dataset, "runs": [run["provenance"] for run in runs], **extra}


def denoising_data(n, split, seed, length=6):
    target = R.text_sequences(n=n, length=length, split=split, seed=seed)
    source = target.copy()
    source[:, 1::3] = 0
    decoder = np.concatenate([np.zeros((n, 1), int), target[:, :-1]], axis=1)
    return source, decoder, target


def compare(configs, labels, task="markov", steps=180, name="comparison"):
    x, y, w = _task(task)
    xt, yt, wt = text_task(task, n=64, seed=73, split="test")
    provenance = {
        **task_provenance(task),
        "evaluation_arrays": [T.array_identity(a) for a in (xt, yt, wt)],
    }
    runs = [
        T.train(
            c,
            x,
            y,
            weights=w,
            steps=steps,
            name="realtext-" + name + "-" + str(i),
            provenance=provenance,
        )
        for i, c in enumerate(configs)
    ]
    losses = [T.loss(r["P"], xt, yt, c, wt) for c, r in zip(configs, runs)]
    return runs, np.array(losses), (x, y, w), (xt, yt, wt)


def counts(P):
    return sum(v.size for v in P.values())


def _compare_panels(
    title,
    configs,
    labels,
    task="markov",
    steps=180,
    caption="Held-out loss is measured on windows from a disjoint real-text partition.",
):
    runs, losses, train, _test = compare(
        configs, labels, task, steps, title.lower().replace(" ", "-")
    )
    panels = (
        matrix(
            "Which tokens can communicate?",
            "Bright cells are permitted query–key pairs.",
            T.attention_mask(train[0].shape[1], configs[0]).astype(float),
        ),
        curve(
            "Training on the same examples",
            "Each line is an actual minibatch training trace.",
            [r["loss"] for r in runs],
            labels,
        ),
        bars(
            "Fresh examples, held out from training",
            caption,
            losses,
            labels,
            ("nats per supervised token", "lower is better"),
        ),
        bars(
            "Count every learned parameter",
            "The comparison includes embeddings and normalization parameters.",
            [counts(r["P"]) for r in runs],
            labels,
            ("Tiny Shakespeare characters", "actual NumPy optimizer updates"),
        ),
    )
    return Study(
        title,
        "Real text · Tiny Shakespeare · measured NumPy training",
        panels,
        run_metadata(runs, task, steps=steps, heldout_loss=losses.tolist(), labels=labels),
    )


def _flash(q, k, v, tile=8):
    n, d = q.shape
    out = np.zeros((n, v.shape[1]))
    history = []
    for i in range(n):
        m, total, acc = -np.inf, 0.0, np.zeros(v.shape[1])
        for start in range(0, n, tile):
            s = k[start : start + tile] @ q[i] / np.sqrt(d)
            new = max(m, float(s.max()))
            factor = np.exp(m - new)
            exp = np.exp(s - new)
            total, acc = total * factor + exp.sum(), acc * factor + exp @ v[start : start + tile]
            m = new
            if i == 0:
                history.append((m, total))
        out[i] = acc / total
    return out, np.array(history)


def _make(name):
    c = T.Config(vocab=len(R.vocabulary()), width=16, heads=4, kv_heads=4, ff=32, max_len=48)
    if name == "causal-vs-bidirectional":
        cs = [c, replace(c, mask="bidirectional")]
        s = _compare_panels(
            "Causal and bidirectional attention", cs, ["causal", "bidirectional"], "fill", 280
        )
        s.panels[3] = matrix(
            "The bidirectional mask",
            "A masked token can use later context; next-token training must hide the answer.",
            T.attention_mask(12, cs[1]).astype(float),
        )
        return s
    if name == "prefix-lm-attention":
        study = _compare_panels(
            "Prefix language-model attention",
            [c, replace(c, mask="prefix", prefix=6), replace(c, mask="bidirectional")],
            ["causal", "prefix", "bidirectional"],
            "copy",
            240,
        )
        study.panels[0] = matrix(
            "A bidirectional observed prefix",
            "The first six observed characters share context; all target suffix slots are masked.",
            T.attention_mask(12, replace(c, mask="prefix", prefix=6)).astype(float),
        )
        study.panels[2].caption = (
            "With one block, prefix-only mask changes cannot alter suffix predictions; "
            "the first two measured losses coincide."
        )
        return study
    if name == "cross-attention":
        cfg = replace(c, cross_attention=True)
        source, x, target = denoising_data(192, "train", 11)
        test_source, test_x, test_target = denoising_data(64, "test", 73)
        run = T.train(
            cfg,
            x,
            target,
            context=source,
            steps=300,
            name="realtext-" + name,
            provenance={
                **task_provenance("denoise", train_seed=11),
                "evaluation_arrays": [
                    T.array_identity(a) for a in (test_source, test_x, test_target)
                ],
            },
        )
        logits, rec = T.forward(run["P"], test_x, cfg, context=test_source, keep=True)
        acc = (logits.argmax(-1) == test_target).mean()
        return Study(
            "Cross attention: read another sequence",
            "Real-text denoising · two masked source characters per six-character span",
            (
                matrix(
                    "Masked source and original text",
                    "Targets are original characters; source positions 2 and 5 are masked.",
                    np.stack([source[0], target[0]]),
                ),
                matrix(
                    "A learned cross-attention head",
                    "Rows are decoder positions; columns are encoder positions.",
                    rec["attention"][1][0, 0],
                ),
                curve(
                    "Train the encoder–decoder link",
                    "The loss is measured while training the projection weights.",
                    run["loss"],
                    ["text denoising loss"],
                ),
                bars(
                    "Held-out token accuracy",
                    "Teacher-forced denoising is evaluated on the disjoint test-text partition.",
                    [acc, 1 / cfg.vocab],
                    ["cross attention", "uniform chance"],
                ),
            ),
            run_metadata(
                [run],
                "denoise",
                accuracy=float(acc),
                evaluation="teacher-forced token accuracy; not free generation",
            ),
        )
    if name == "multi-head-vs-single":
        return multi_relation_heads()
    if name == "grouped-query-attention":
        cs = [replace(c, heads=4, kv_heads=n) for n in (4, 2, 1)]
        s = _compare_panels("Grouped-query attention", cs, ["MHA", "GQA", "MQA"], steps=240)
        s.panels[0] = bars(
            "KV cache bytes per token",
            "Keys and values share heads; queries remain separate.",
            [2 * 32 * n * 128 * 2 for n in (8, 2, 1)],
            ["8 KV heads", "2 KV heads", "1 KV head"],
            ("32 layers, head width 128", "FP16: 2 bytes per value"),
        )
        return s
    if name in {"kv-cache", "attention-sinks", "residual-stream"}:
        from . import gpt2, gpt2_hooks

        tok = gpt2.Tokenizer()
        ids = tok.encode(R.text("test")[:64])
        if name == "kv-cache":

            def build():
                full = gpt2_hooks.forward(ids)["logits"]
                cache, rows = None, []
                for token in ids:
                    r = gpt2_hooks.forward([token], cache=cache)
                    rows.append(r["logits"][0])
                    cache = r["cache"]
                return {"diff": np.max(np.abs(full - np.array(rows)), axis=1)}

            a = cached_npz(
                "next-kv-cache",
                {
                    "ids": ids,
                    "version": 2,
                    "source": R.source_hash("next_attention.py", "gpt2_hooks.py"),
                    "model_revision": gpt2.REVISION,
                },
                build,
            )
            n = np.arange(1, len(ids) + 1)
            return Study(
                "The key–value cache",
                "GPT-2 small · actual cached and full forwards",
                (
                    curve(
                        "Cache equivalence, token by token",
                        "Cached decoding and the full forward agree to floating-point precision.",
                        a["diff"],
                        ["max logit error"],
                    ),
                    curve(
                        "Projection work at each step",
                        "Without caching, all previous tokens are projected again.",
                        [n, np.ones(len(n))],
                        ["uncached tokens", "cached tokens"],
                        x=n,
                    ),
                    curve(
                        "Cache memory grows linearly",
                        "Each token stores a key and value in all twelve layers.",
                        n[None] * 2 * 12 * 768 * 4,
                        ["cache bytes"],
                        x=n,
                    ),
                    bars(
                        "Total projected token positions",
                        "Caching avoids recomputing previous keys and values.",
                        [n.sum(), len(n)],
                        ["uncached", "cached"],
                        ("same logits", "attention still reads the cache"),
                    ),
                ),
            )
        if name == "residual-stream":
            tr = gpt2.run(tok.decode(ids))
            ao = np.linalg.norm(tr.attn_out[:, -1], axis=1)
            mo = np.linalg.norm(tr.mlp_out[:, -1], axis=1)
            lens = gpt2.logit_lens(tr)
            target = int(lens[-1].argmax())
            return Study(
                "The residual stream",
                "GPT-2 small · measured writes at the last token",
                (
                    curve(
                        "The stream accumulates layer writes",
                        "Each block adds an attention write and an MLP write to the shared stream.",
                        [ao, mo],
                        ["attention norm", "MLP norm"],
                        x=np.arange(1, 13),
                    ),
                    curve(
                        "A logit lens at every write",
                        "Read the final predicted token through the fixed output projection.",
                        lens[:, target],
                        [repr(tok.decode([target]))],
                    ),
                    matrix(
                        "Attention writes to the stream",
                        "Each row is a layer; the first 48 channels are shown.",
                        tr.attn_out[:, -1, :48],
                    ),
                    matrix(
                        "MLP writes to the stream",
                        "The learned writes can reinforce or oppose existing stream components.",
                        tr.mlp_out[:, -1, :48],
                    ),
                ),
            )

        # Stream real text, preserving absolute positions after every eviction.
        text = R.text("test")[:520]
        stream = tok.encode(text)[:64]
        window = 32

        def build_sinks():
            losses = []
            masses = []
            for mode in ("full", "window", "sinks"):
                cache = None
                kept = []
                nll = []
                for i, token in enumerate(stream[:-1]):
                    if cache is not None and mode != "full" and len(kept) >= window:
                        keep = list(range(len(kept) - window + 1, len(kept)))
                        if mode == "sinks":
                            keep = list(range(4)) + list(range(len(kept) - window + 5, len(kept)))
                        cache = [(k[:, keep], v[:, keep]) for k, v in cache]
                        kept = [kept[j] for j in keep]
                    result = gpt2_hooks.forward(
                        [token], cache=cache, position_ids=[i], record=(mode == "full")
                    )
                    cache = result["cache"]
                    kept.append(i)
                    logit = result["logits"][0].astype(float)
                    nll.append(np.logaddexp.reduce(logit) - logit[stream[i + 1]])
                    if mode == "full" and i >= window:
                        masses.append(result["attn"][:, :, 0, 0])
                losses.append(nll)
            return {"mass": np.mean(masses, axis=0), "nll": np.asarray(losses)}

        a = cached_npz(
            "next-attention-sinks",
            {
                "ids": stream,
                "window": window,
                "version": 4,
                "source": R.source_hash("next_attention.py", "gpt2_hooks.py"),
                "model_revision": gpt2.REVISION,
            },
            build_sinks,
        )
        nll = np.asarray(a["nll"])
        perplexity = np.exp(nll[:, window:].mean(1))
        running = np.exp(
            np.cumsum(nll[:, window:], axis=1) / np.arange(1, nll.shape[1] - window + 1)
        )
        return Study(
            "Attention sinks",
            "GPT-2 small · streamed text, fixed absolute positions, 32 cache slots",
            (
                matrix(
                    "Attention paid to position zero",
                    "Each cell averages attention on token zero over all full-cache late queries.",
                    a["mass"],
                ),
                curve(
                    "Streaming perplexity after eviction",
                    "All policies predict the same next tokens; both limited caches use 32 slots.",
                    running,
                    ["full cache", "last 32", "4 sinks + last 28"],
                    x=np.arange(window, nll.shape[1]),
                ),
                bars(
                    "Perplexity on the same suffix",
                    "Keep four sink tokens while preserving the original positional embeddings.",
                    perplexity,
                    ["full cache", "last 32", "4 sinks + last 28"],
                    ("Lower perplexity is better.", "64 real-text tokens; no universal claim."),
                ),
                bars(
                    "Memory used by each policy",
                    "Sink retention spends four existing slots; it does not grow the window.",
                    [len(stream) - 1, window, window],
                    ["full cache", "last 32", "4 sinks + last 28"],
                    ("Slots hold both keys and values in every layer.",),
                ),
            ),
            {
                "scope": "streamed held-out next-token perplexity",
                "tokens": len(stream),
                "window": window,
                "perplexity": perplexity.tolist(),
            },
        )
    if name == "sliding-window-attention":
        cs = [replace(c, mask="window", window=4, layers=n) for n in (1, 2, 3)]
        s = _compare_panels(
            "Sliding-window attention",
            cs,
            ["1 layer", "2 layers", "3 layers"],
            "markov",
            240,
            "Next-character loss on real text; depth changes the structurally reachable context.",
        )
        masks = []
        reach = np.eye(12, dtype=bool)
        for _ in range(3):
            reach = (reach.astype(int) @ T.attention_mask(12, cs[0]).astype(int)) > 0
            masks.append(reach.copy())
        s.panels[0] = matrix(
            "Reach after three local layers",
            "A width-four mask reaches at most nine previous positions after three layers.",
            masks[-1].astype(float),
        )
        return s
    if name == "sparse-attention-patterns":
        return sparse_patterns()
    if name == "linear-attention":
        s = _compare_panels(
            "Linear attention",
            [c, replace(c, attention="linear")],
            ["softmax", "ELU+1 kernel"],
            steps=240,
        )
        rng = np.random.default_rng(8)
        q, k, v = [rng.normal(size=(16, 8)) for _ in range(3)]

        def phi(z):
            return np.where(z > 0, z + 1, np.exp(np.minimum(z, 0)))

        a = phi(q) @ phi(k).T
        out = (a @ v) / a.sum(1, keepdims=True)
        rec = (phi(q) @ (phi(k).T @ v)) / (phi(q) @ phi(k).sum(0))[:, None]
        sm = np.exp(q @ k.T / np.sqrt(8))
        sm /= sm.sum(1, keepdims=True)
        s.panels[0] = bars(
            "Associativity is exact; softmax differs",
            "The kernel recurrence equals kernel attention, but does not equal softmax attention.",
            [np.max(abs(out - rec)), np.max(abs(out - sm @ v))],
            ["kernel recurrence error", "difference from softmax"],
        )
        return s
    if name == "flash-attention-tiling":
        rng = np.random.default_rng(4)
        q, k, v = [rng.normal(size=(32, 16)) for _ in range(3)]
        scores = q @ k.T / 4
        a = np.exp(scores - scores.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        out, hist = _flash(q, k, v)
        ns = np.array([64, 128, 256, 512, 1024, 2048])
        return Study(
            "Flash attention: the online softmax",
            "Exact NumPy arithmetic · 8-key tiles",
            (
                matrix(
                    "The naive attention matrix",
                    "Softmax combines every query–key pair before multiplying values.",
                    a,
                ),
                curve(
                    "Rescale whenever the maximum changes",
                    "A running maximum and sum keep the tile calculation stable.",
                    hist.T,
                    ["running maximum", "rescaled sum"],
                ),
                bars(
                    "The same answer, tile by tile",
                    "Online softmax is an exact reformulation up to floating-point rounding.",
                    [float(np.max(abs(a @ v - out))), float(np.linalg.norm(a @ v - out))],
                    ["maximum error", "L2 error"],
                ),
                curve(
                    "Scratch storage, in float64 bytes",
                    "Dense scores need n² entries; a fixed tile needs only n×8 entries.",
                    [8 * ns * ns, 8 * ns * 8],
                    ["dense scores", "one score tile"],
                    x=ns,
                ),
            ),
        )
    if name == "attention-temperature":
        rng = np.random.default_rng(54)
        ds = np.array([8, 16, 32, 64, 128, 256, 512])
        h = []
        hs = []
        g = []
        for d in ds:
            q = rng.normal(size=(128, d))
            k = rng.normal(size=(32, d))
            s = q @ k.T
            p = np.exp(s - s.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            z = s / np.sqrt(d)
            p2 = np.exp(z - z.max(1, keepdims=True))
            p2 /= p2.sum(1, keepdims=True)
            h.append(-(p * np.log(p + 1e-30)).sum(1).mean())
            hs.append(-(p2 * np.log(p2 + 1e-30)).sum(1).mean())
            g.append((p * (1 - p)).sum(1).mean())
        cs = [c, replace(c, scale_scores=False)]
        s = _compare_panels("Why divide scores by √d?", cs, ["scaled", "unscaled"], steps=220)
        s.panels[0] = curve(
            "Entropy as head width increases",
            "Unscaled dot products increasingly saturate the softmax.",
            [h, hs],
            ["unscaled entropy", "scaled entropy"],
            x=ds,
        )
        s.panels[3] = curve(
            "Softmax Jacobian trace",
            "The sum p(1−p) falls as random logits saturate.",
            [g],
            ["unscaled Jacobian trace"],
            x=ds,
        )
        return s
    if name == "pre-ln-vs-post-ln":
        configurations = [
            replace(c, width=8, ff=16, heads=2, kv_heads=2, layers=depth, placement=place)
            for depth in (4, 12, 24)
            for place in ("pre", "post")
        ]
        labels = ["%s-LN, %d layers" % (cfg.placement, cfg.layers) for cfg in configurations]
        runs, losses, _, _ = compare(configurations, labels, steps=100, name="pre-post-depth-sweep")
        x, y, weights = _task(n=12)
        gradients = []
        for cfg in configurations[-2:]:
            _, grad = T.loss_grads(T.init(cfg), x, y, cfg, weights)
            gradients.append([np.linalg.norm(grad["l%d.Q" % i]) for i in range(cfg.layers)])
        return Study(
            "Pre-LN and post-LN with depth",
            "Real text; 4, 12 and 24 transformer blocks",
            (
                curve(
                    "Gradients through 24 blocks",
                    "Measure each query projection before the first optimizer update.",
                    gradients,
                    ["pre-LN", "post-LN"],
                    x=np.arange(1, 25),
                ),
                curve(
                    "Four-layer training",
                    "Normalization placement is the only changed setting.",
                    [r["loss"] for r in runs[:2]],
                    labels[:2],
                ),
                curve(
                    "Twelve and twenty-four layers",
                    "These short seeded runs report measured behaviour without assuming a winner.",
                    [r["loss"] for r in runs[2:]],
                    labels[2:],
                ),
                bars(
                    "Held-out loss at every depth",
                    "All models are evaluated on the same disjoint test-text windows.",
                    losses,
                    labels,
                ),
            ),
            run_metadata(runs),
        )
    if name == "rmsnorm-vs-layernorm":
        return _compare_panels(
            "RMSNorm and LayerNorm",
            [c, replace(c, norm="rms")],
            ["LayerNorm", "RMSNorm"],
            steps=240,
        )
    if name == "swiglu-mlp":
        return _compare_panels(
            "A gated transformer MLP",
            [replace(c, ff=48), replace(c, ff=32, activation="swiglu")],
            ["GELU", "SwiGLU"],
            steps=240,
        )
    if name == "weight-tying":
        cs = [c, replace(c, tied=True)]
        s = _compare_panels("Reuse the input embedding", [*cs], ["untied", "tied"], steps=240)
        s.panels[0] = bars(
            "GPT-2’s shared embedding table",
            "One table maps tokens into vectors and vectors back into vocabulary logits.",
            [50257 * 768, 50257 * 768 * 4],
            ["saved parameters", "saved FP32 bytes"],
        )
        return s
    if name == "parallel-block":
        return _compare_panels(
            "Sequential and parallel blocks",
            [c, replace(c, parallel=True)],
            ["sequential", "parallel"],
            steps=240,
            caption="Both branches read the same normalized input; "
            "this experiment measures the loss trade-off.",
        )
    if name == "encoder-decoder-family":
        return encoder_decoder_family()
    if name == "transformer-depth-vs-width":
        cs = [
            replace(c, width=36, ff=67, layers=2),
            replace(c, width=28, ff=35, layers=4),
            replace(c, width=12, ff=88, layers=8),
        ]
        return _compare_panels(
            "Transformer depth and width",
            cs,
            ["wide × 2", "medium × 4", "narrow × 8"],
            "markov",
            260,
            caption="Real-text next-character loss; parameter counts are reported "
            "rather than assumed equal.",
        )
    raise KeyError(name)


def multi_relation_heads():
    configs = [
        T.Config(
            vocab=len(R.vocabulary()), width=32, heads=h, kv_heads=h, ff=64, layers=2, max_len=16
        )
        for h in (1, 8)
    ]
    x, y, weights = text_task(n=384)
    xt, yt, wt = text_task(n=128, seed=73, split="test")
    runs = [
        T.train(
            c,
            x,
            y,
            weights=weights,
            steps=900,
            name="realtext-heads-%d" % c.heads,
            provenance={
                **task_provenance(),
                "evaluation_arrays": [T.array_identity(a) for a in (xt, yt, wt)],
            },
        )
        for c in configs
    ]
    losses = [T.loss(r["P"], xt, yt, c, wt) for r, c in zip(runs, configs)]
    logits, records = T.forward(runs[1]["P"], xt, configs[1], keep=True)
    attn = records["attention"][-1]
    positions = np.arange(1, xt.shape[1])
    previous = attn[:, :, positions, positions - 1].mean((0, 2))
    matching_mask = (xt[:, :, None] == xt[:, None, :]) & np.tril(
        np.ones((xt.shape[1], xt.shape[1]), bool), -1
    )
    matching = (attn * matching_mask[:, None]).sum(-1).mean((0, 2))
    accuracy = (logits.argmax(-1) == yt).mean()
    return Study(
        "One wide head or eight narrow heads",
        "Real-text next-character prediction; equal parameter counts",
        (
            matrix(
                "Observed characters and next-character targets",
                "Rows are input and next-character IDs from a held-out Shakespeare window.",
                np.stack([xt[0], yt[0]]),
            ),
            curve(
                "Train on the same real-text windows",
                "Both models have width 32 and two blocks; only the head partition changes.",
                [r["loss"] for r in runs],
                ["one head", "eight heads"],
            ),
            matrix(
                "Two observed attention summaries",
                "Eight heads: mass on the previous position and on earlier matching characters.",
                np.column_stack([previous, matching]),
                (
                    "Eight-head token accuracy: %.1f%%" % (100 * accuracy),
                    "These overlapping summaries do not assign causal roles to heads.",
                ),
            ),
            bars(
                "Held-out next-character loss",
                "A different head partition need not improve this small real-text experiment.",
                losses,
                ["one head", "eight heads"],
                ("Parameters each: %d" % counts(runs[0]["P"]),),
            ),
        ),
        run_metadata(runs, parameters=[counts(r["P"]) for r in runs], accuracy=float(accuracy)),
    )


def _hidden(P, x, cfg, prefix="", context=None, record=None):
    """A differentiable transformer stack whose hidden state can feed another stack."""
    h = P[prefix + "E"][x] + P[prefix + "pos"][np.arange(x.shape[1])]
    params = {k.removeprefix(prefix): v for k, v in P.items() if k.startswith(prefix)}
    mask = T.attention_mask(x.shape[1], cfg)
    attn = []
    for layer in range(cfg.layers):
        key = "l%d." % layer
        z = T._norm(h, params, key + "n1", cfg)
        h = h + T._attention(z, z, params, key, cfg, mask, attn)
        if context is not None:
            z = T._norm(h, params, key + "nc", cfg)
            h = h + T._attention(
                z,
                context,
                params,
                key + "cross.",
                cfg,
                np.ones((x.shape[1], context.v.shape[1]), bool),
                attn,
            )
        z = T._norm(h, params, key + "n2", cfg)
        h = h + T.gelu(z @ params[key + "W1"]) @ params[key + "W2"]
    if record is not None:
        record.extend(attn)
    return T._norm(h, params, "final", cfg)


def encoder_decoder_loss(P, source, decoder, target, enc, dec, gradients=True):
    params = {k: T.Var(v) for k, v in P.items()}
    context = _hidden(params, source, enc, "encoder.")
    hidden = _hidden(params, decoder, dec, "decoder.", context=context)
    logits = hidden @ params["decoder.U"]
    shifted = logits - logits.v.max(-1, keepdims=True)
    logp = shifted - shifted.exp().sum(-1, True).log()
    selected = logp[np.arange(len(target))[:, None], np.arange(target.shape[1]), target]
    objective = -selected.mean()
    if not gradients:
        return float(objective.v), logits.v
    objective.backward()
    return float(objective.v), {
        k: np.zeros_like(v.v) if v.g is None else v.g for k, v in params.items()
    }


def encoder_decoder_family():
    from .optim import Adam

    enc = T.Config(
        vocab=len(R.vocabulary()),
        width=16,
        heads=4,
        kv_heads=4,
        ff=32,
        layers=1,
        max_len=16,
        mask="bidirectional",
    )
    dec = replace(enc, mask="causal", cross_attention=True)
    training = denoising_data(256, "train", 81)
    testing = denoising_data(64, "test", 73)
    source, shifted, target = [np.concatenate([a, b]) for a, b in zip(training, testing)]
    # All three families restore the same real text with explicitly different conditioning.
    encoder_input = np.concatenate([source, np.zeros_like(source)], axis=1)
    encoder_target = np.concatenate([np.zeros_like(source), target], axis=1)
    w = np.zeros_like(encoder_target, float)
    w[:, 6:] = 1
    decoder_input = np.concatenate([source, shifted], axis=1)
    causal = replace(enc, mask="causal")
    baseline = [
        T.train(
            cfg,
            xx[:256],
            encoder_target[:256],
            weights=w[:256],
            steps=420,
            name="realtext-family-" + kind,
            provenance={
                **task_provenance("denoise", train_seed=81),
                "evaluation_arrays": [T.array_identity(a) for a in testing],
            },
        )
        for cfg, xx, kind in ((enc, encoder_input, "encoder"), (causal, decoder_input, "decoder"))
    ]

    def build():
        P = {"encoder." + k: v for k, v in T.init(enc, 2).items()}
        P.update({"decoder." + k: v for k, v in T.init(dec, 3).items()})
        opt = Adam(P, 0.003)
        rng = np.random.default_rng(82)
        losses = []
        for _ in range(420):
            ix = rng.choice(256, 24, replace=False)
            loss, grads = encoder_decoder_loss(P, source[ix], shifted[ix], target[ix], enc, dec)
            norm = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
            opt.step(P, {k: g * min(1.0, 1.0 / max(norm, 1e-12)) for k, g in grads.items()})
            losses.append(loss)
        loss, logits = encoder_decoder_loss(
            P, source[256:], shifted[256:], target[256:], enc, dec, False
        )
        return dict(
            **{"P_" + k: v for k, v in P.items()},
            loss=np.asarray(losses),
            heldout=loss,
            accuracy=(logits.argmax(-1) == target[256:]).mean(),
        )

    joint_settings = dict(
        version=2,
        steps=420,
        seed=82,
        encoder_seed=2,
        decoder_seed=3,
        encoder=asdict(enc),
        decoder=asdict(dec),
        optimizer="Adam",
        lr=0.003,
        batch=24,
        clip=1.0,
        inputs=[T.array_identity(v) for v in (source, shifted, target)],
        sources=T.training_sources("next_attention.py", "realdata.py"),
    )
    joint = cached_npz("next-realtext-joint-encoder-decoder", joint_settings, build)
    joint_evidence = R.save_evidence(
        "attention-joint-encoder-decoder",
        {k[2:]: v for k, v in joint.items() if k.startswith("P_")},
        {k: joint[k] for k in ("loss", "heldout", "accuracy")},
        task_provenance("denoise", train_seed=81),
        joint_settings,
        ("next_attention.py", "tinylm.py", "optim.py"),
    )
    held = [
        T.loss(r["P"], xx[256:], encoder_target[256:], cfg, w[256:])
        for r, xx, cfg in zip(baseline, (encoder_input, decoder_input), (enc, causal))
    ]
    held.append(float(joint["heldout"]))
    return Study(
        "Encoder, decoder, encoder–decoder",
        "Three trained families restore masked real-text spans; conditioning differs",
        (
            matrix(
                "Encoder: masked target slots",
                "Bidirectional attention sees the source and six blank target positions.",
                T.attention_mask(12, enc).astype(float),
            ),
            matrix(
                "Decoder: teacher-forced targets",
                "A causal stack sees the source followed by shifted target symbols.",
                T.attention_mask(12, causal).astype(float),
            ),
            curve(
                "Encoder–decoder: train both stacks",
                "The trained encoder passes contextual hidden states to decoder cross attention.",
                [baseline[0]["loss"], baseline[1]["loss"], joint["loss"]],
                ["encoder masked slots", "decoder next token", "joint encoder–decoder"],
            ),
            bars(
                "Held-out target loss",
                "All restore the original text; decoder models also receive "
                "shifted target prefixes.",
                held,
                ["encoder", "decoder", "encoder–decoder"],
                ("Joint target accuracy: %.1f%%" % (100 * joint["accuracy"]),),
            ),
        ),
        run_metadata(
            baseline,
            "denoise",
            joint_run=joint_evidence,
            comparison="Different conditioning and parameter counts; "
            "not a controlled architecture ranking",
        ),
    )


def sparse_mask(n, kind, window=4):
    """Seeded causal sparse patterns; each row always includes its own token."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    local = (j <= i) & (i - j < window)
    if kind == "dense":
        return j <= i
    if kind == "strided":
        return (j <= i) & ((i - j < window) | (j % window == 0))
    block = (j <= i) & (i // window == j // window)
    if kind == "block":
        return block
    if kind == "global":
        return (j <= i) & (local | (j == 0))
    if kind == "random":
        # Deterministic BigBird-style local + first global + random prior links.
        rng = np.random.default_rng(731)
        result = local | (j == 0)
        for row in range(n):
            result[row, rng.choice(row + 1, min(2, row + 1), replace=False)] = True
        return result
    raise ValueError(kind)


def _pattern_counts(n, kind, window=32):
    """Exact counts without allocating a large n-by-n mask."""
    if n <= 256:
        return int(sparse_mask(n, kind, window).sum())
    total = 0
    rng = np.random.default_rng(731)
    for row in range(n):
        if kind == "dense":
            total += row + 1
            continue
        local = set(range(max(0, row - window + 1), row + 1))
        if kind == "strided":
            total += len(local) + len(range(0, max(0, row - window + 1), window))
        elif kind == "block":
            total += row % window + 1
        elif kind == "global":
            total += len(local | {0})
        elif kind == "random":
            total += len(local | {0} | set(rng.choice(row + 1, min(2, row + 1), False)))
        else:
            raise ValueError(kind)
    return total


def sparse_patterns():
    c = T.Config(vocab=len(R.vocabulary()), width=16, heads=4, kv_heads=4, ff=32, max_len=48)
    kinds = ("dense", "strided", "block", "global", "random")
    x, y, weights = _task()
    xt, yt, wt = text_task(n=64, seed=73, split="test")
    masks = [sparse_mask(12, kind) for kind in kinds]
    runs = [
        T.train(
            c,
            x,
            y,
            weights=weights,
            steps=200,
            seed=50,
            name="realtext-sparse-" + kind,
            mask=mask,
            provenance={
                **task_provenance(),
                "evaluation_arrays": [T.array_identity(a) for a in (xt, yt, wt)],
            },
        )
        for kind, mask in zip(kinds, masks)
    ]
    heldout = [T.loss(run["P"], xt, yt, c, wt, mask=mask) for run, mask in zip(runs, masks)]

    def build_counts():
        ns = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536])
        counts = [[_pattern_counts(int(n), kind) for n in ns] for kind in kinds]
        return dict(n=ns, counts=np.array(counts))

    d = cached_npz(
        "next-sparse-pattern-counts",
        {"version": 2, "source": R.source_hash("next_attention.py")},
        build_counts,
    )
    return Study(
        "Sparse attention patterns",
        "Five causal masks trained on the same real-text next-character task",
        (
            matrix(
                "Five wiring patterns",
                "Dense, strided, block-local, global-token and random-local masks, stacked.",
                np.concatenate(masks, axis=0).astype(float),
            ),
            curve(
                "Train under each mask",
                "Only permitted query–key links participate in the forward and backward passes.",
                [run["loss"] for run in runs],
                list(kinds),
            ),
            bars(
                "Held-out next-character loss",
                "The loss is measured on disjoint real-text windows; no mask is assumed to win.",
                heldout,
                list(kinds),
            ),
            curve(
                "Count actual allowed pairs",
                "Fixed-window block, global and random patterns remain linear in sequence length.",
                d["counts"],
                list(kinds),
                x=d["n"],
            ),
        ),
        run_metadata(runs, heldout_loss=heldout),
    )


@cache
def make(name):
    """Build the requested study with explicit per-panel evidence categories."""
    study = _make(name)
    measured, analytic = "trained_model_measurement", "analytic"
    kinds = [analytic, measured, measured, analytic]
    if name in {"kv-cache", "attention-sinks", "residual-stream"}:
        from . import gpt2

        study.metadata.update(
            evidence_kind="pretrained_model_measurement",
            pretrained_model="GPT-2 small",
            model_revision=gpt2.REVISION,
            dataset=R.text_metadata("test"),
            pretraining_overlap="Unknown; this partition is held out only from our local training",
        )
        kinds = {
            "kv-cache": ["pretrained_model_measurement", analytic, analytic, analytic],
            "attention-sinks": ["pretrained_model_measurement"] * 3 + [analytic],
            "residual-stream": ["pretrained_model_measurement"] * 4,
        }[name]
    elif name == "flash-attention-tiling":
        kinds = ["numerical_identity"] * 3 + [analytic]
        study.metadata.update(
            evidence_kind="numerical_identity",
            seed=4,
            limitation="Random arrays verify arithmetic; no training or speed benchmark",
        )
    else:
        study.metadata.setdefault("evidence_kind", measured)
        study.metadata.setdefault(
            "limitations",
            [
                "Small fixed-budget real-text experiments; outcomes may include "
                "underfitting or overfitting",
                "No universal architecture ranking or convergence claim",
            ],
        )
        if name == "causal-vs-bidirectional":
            kinds[-1] = analytic
        if name == "cross-attention":
            kinds = ["dataset_example", measured, measured, measured]
        if name == "multi-head-vs-single":
            kinds = ["dataset_example", measured, measured, measured]
        if name == "pre-ln-vs-post-ln":
            kinds = ["untrained_model_measurement", measured, measured, measured]
        if name == "encoder-decoder-family":
            kinds = [analytic, analytic, measured, measured]
        if name == "attention-temperature":
            kinds = ["random_array_simulation", measured, measured, "random_array_simulation"]
            study.metadata["numerical_example"] = dict(
                seed=54, distribution="independent standard normal Q and K"
            )
        if name == "linear-attention":
            kinds[0] = "numerical_identity"
            study.metadata["numerical_example"] = dict(
                seed=8, distribution="independent standard normal Q, K, V"
            )
    study.metadata["panel_evidence"] = kinds
    study.metadata["source_sha256"] = R.source_hash("next_attention.py", "tinylm.py", "realdata.py")
    for panel, kind in zip(study.panels, kinds):
        label = kind.replace("_", " ")
        panel.notes = tuple(panel.notes) + ("Evidence: " + label,)
    return study.validate()

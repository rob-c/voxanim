"""Redesigned experiments for animations 1-6 of the "next 100" (attention, how it is wired).

Two of the first experiments (:mod:`voxanim.next_attention`) could not show their concept:

* ``prefix-lm-attention`` trained one-block models with every reply slot blanked. In one block
  the reply reads the prompt's characters directly, so what the prompt's rows may read never
  reaches it: the causal and prefix masks gave the same loss to the last digit (3.714 / 3.714).
* ``multi-head-vs-single`` trained on 384 windows for 900 updates: both models learned the
  windows by heart (held-out loss above even guessing) and no head differed from an even spread.

Their replacements are here. Both draw every training batch fresh from the training part of
Tiny Shakespeare (no window is seen twice, so nothing can be memorised), train three seeds of
every model and measure on windows of the disjoint test part.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import numpy as np

from . import realdata as R
from . import tinylm as T
from .cache import cached_npz
from .next_attention import bars, counts, curve, matrix
from .optim import AdamW
from .study import Study

VERSION = 1
SEEDS = (0, 1, 2)
BATCH, LR, CLIP = 32, 0.003, 1.0
HELD_OUT, HELD_OUT_SEED = 512, 73


def _windows(split, length, n=HELD_OUT, seed=HELD_OUT_SEED):
    """``n`` distinct windows of ``length`` characters from one part of the text."""
    return R.text_sequences(n=n, length=length, split=split, seed=seed)


def _fit(name, config, batch_of, steps, seed, settings, probe=None, every=0):
    """Train ``config`` for ``steps`` AdamW updates on batches ``batch_of(starts)`` cut from
    the training text at fresh random places, the gradient clipped to norm 1 as
    :func:`voxanim.tinylm.train` does. ``probe(P)`` (a dict of numbers) is recorded every
    ``every`` updates. Cached by every setting."""
    settings = dict(settings, config=asdict(config), steps=steps, seed=seed, batch=BATCH, lr=LR,
                    clip=CLIP, every=every, version=VERSION, optimizer="AdamW",
                    sources=T.training_sources(), train=R.text_metadata("train")["split_sha256"],
                    test=R.text_metadata("test")["split_sha256"])

    def build():
        text = R.encode(R.text("train"))
        P, rng = T.init(config, seed), np.random.default_rng(seed + 100)
        opt = AdamW(P, LR)
        losses, probes = [], {}
        for step in range(steps):
            x, y, w = batch_of(text, rng)
            value, G = T.loss_grads(P, x, y, config, w)
            norm = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
            ratio = min(1.0, CLIP / max(norm, 1e-12))
            opt.step(P, {k: g * ratio for k, g in G.items()})
            losses.append(value)
            if probe is not None and every and (step + 1) % every == 0:
                for key, v in probe(P).items():
                    probes.setdefault(key, []).append(v)
        return {**{"P_" + k: v for k, v in P.items()}, "loss": np.array(losses),
                **{"probe_" + k: np.array(v) for k, v in probes.items()}}

    a = cached_npz("study-attention-a-" + name, settings, build)
    return {"P": {k[2:]: v for k, v in a.items() if k.startswith("P_")}, "loss": a["loss"],
            "probes": {k[6:]: v for k, v in a.items() if k.startswith("probe_")},
            "settings": settings}


def _provenance(task, **extra):
    return {
        "dataset": R.text_metadata("train"),
        "evaluation_dataset": R.text_metadata("test"),
        "task": task,
        "source_sha256": R.source_hash("study_attention_a.py", "realdata.py"),
        "training_windows": "every batch cut at fresh random places in the training part",
        "heldout_windows": "%d from the test part (seed %d), without replacement" % (
            HELD_OUT, HELD_OUT_SEED),
        "seeds": list(SEEDS),
        "selection": "fixed settings; no checkpoint or seed selection",
        **extra,
    }


# ------------------------------------------------------------------ 2 prefix-lm-attention
PROMPT = 8                                  # characters in the prompt, and in the reply
PREFIX_STEPS, PREFIX_EVERY = 2000, 250
MASKS = ("causal", "prefix", "bidirectional")


def sorting_rows(prompts):
    """The copy-and-transform task as token rows. The reply is the prompt's characters in
    character order (line break, space, punctuation, capitals, lower case). The row is the
    prompt, a start mark (0) and the reply so far; the targets are the reply, scored on its
    slots only: ``(x, y, weights)``."""
    prompts = np.asarray(prompts, int)
    reply = np.sort(prompts, 1)
    start = np.zeros((len(prompts), 1), int)
    x = np.concatenate([prompts, start, reply[:, :-1]], 1)
    y = np.concatenate([np.zeros_like(prompts), reply], 1)
    w = np.concatenate([np.zeros(prompts.shape), np.ones(prompts.shape)], 1)
    return x, y, w


def write_reply(P, config, prompts):
    """The reply the model writes on its own, one character at a time: each slot's most
    likely character is fed back as the next slot's input. Slots not yet written hold the
    blank (0) - there is nothing else to put there."""
    prompts = np.asarray(prompts, int)
    n, k = prompts.shape
    x = np.zeros((n, 2 * k), int)
    x[:, :k] = prompts
    out = np.zeros((n, k), int)
    for i in range(k):
        out[:, i] = T.forward(P, x, config)[:, k + i].argmax(-1)
        if i < k - 1:
            x[:, k + 1 + i] = out[:, i]
    return out


def prefix_config(mask):
    return T.Config(vocab=len(R.vocabulary()), width=32, heads=4, kv_heads=4, ff=64, layers=2,
                    max_len=2 * PROMPT, mask=mask, prefix=PROMPT)


def prefix_heldout():
    """The held-out prompts and their rows: ``(prompts, x, y, weights)``."""
    prompts = _windows("test", PROMPT)
    return (prompts, *sorting_rows(prompts))


@lru_cache(maxsize=1)
def prefix_runs():
    """``{mask: [run per seed]}``: each run's parameters, training trace, the held-out
    written accuracy every 250 updates, and its final held-out scores."""
    prompts, xt, yt, wt = prefix_heldout()
    target = yt[:, PROMPT:]

    def batch_of(text, rng):
        starts = rng.integers(0, len(text) - PROMPT + 1, BATCH)
        return sorting_rows(text[starts[:, None] + np.arange(PROMPT)])

    out = {}
    for mask in MASKS:
        config = prefix_config(mask)
        runs = []
        for seed in SEEDS:
            run = _fit("prefix-" + mask, config, batch_of, PREFIX_STEPS, seed,
                       {"task": "sort the prompt", "prompt": PROMPT,
                        "heldout": T.array_identity(xt)},
                       lambda P: {"written": float(
                           (write_reply(P, config, prompts) == target).mean())},  # noqa: B023
                       PREFIX_EVERY)
            written = write_reply(run["P"], config, prompts)
            run.update(config=config, written=written,
                       accuracy=float((written == target).mean()),
                       exact=float((written == target).all(1).mean()),
                       heldout_loss=T.loss(run["P"], xt, yt, config, wt))
            runs.append(run)
        out[mask] = runs
    return out


def prefix_lm_attention():
    runs = prefix_runs()
    prompts, xt, _yt, _wt = prefix_heldout()
    table = lambda key: np.array([[r[key] for r in runs[m]] for m in MASKS])  # noqa: E731
    accuracy, exact, loss = table("accuracy"), table("exact"), table("heldout_loss")
    curves = np.array([[r["probes"]["written"] for r in runs[m]] for m in MASKS])
    updates = np.arange(1, curves.shape[2] + 1) * PREFIX_EVERY
    spread = lambda a: ["%s: %.3f to %.3f over %d seeds" % (m, row.min(), row.max(), len(row))  # noqa: E731
                        for m, row in zip(MASKS, a)]
    return Study(
        "The prefix mask",
        "Real text · Tiny Shakespeare · two-block NumPy transformers, three seeds a mask",
        (
            matrix(
                "The prefix mask",
                "Rows read columns: the eight prompt characters read each other both ways; "
                "the start mark and the reply read only leftwards.",
                T.attention_mask(2 * PROMPT, prefix_config("prefix")).astype(float),
            ),
            curve(
                "Held-out replies written correctly, as training goes on",
                "Share of reply characters right when each model writes the reply itself; "
                "mean of three seeds.",
                curves.mean(1), list(MASKS), x=updates,
            ),
            bars(
                "Characters right in replies the model writes itself",
                "512 held-out prompts; each written character is fed back as the next input.",
                accuracy.mean(1), list(MASKS), spread(accuracy),
            ),
            bars(
                "Held-out loss when shown the true reply so far",
                "The bidirectional mask lets each reply slot read the next slot's input, "
                "which is its own answer: its low loss is copying, not sorting.",
                loss.mean(1), list(MASKS), spread(loss),
            ),
        ),
        _provenance(
            "write the prompt's eight characters in character order, after the prompt",
            labels=list(MASKS), steps=PREFIX_STEPS, prompt=PROMPT, updates=updates.tolist(),
            configs=[asdict(prefix_config(m)) for m in MASKS],
            parameters=counts(runs["causal"][0]["P"]),
            written_accuracy=accuracy.tolist(), exact_match=exact.tolist(),
            heldout_loss=loss.tolist(), written_accuracy_curves=curves.tolist(),
            training_loss=[[r["loss"].tolist() for r in runs[m]] for m in MASKS],
            heldout_arrays=[T.array_identity(a) for a in (prompts, xt)],
            runs=[{"settings": dict(r["settings"].items())}
                  for m in MASKS for r in runs[m]],
            designs_tried=[
                "first study: one block, every reply slot blanked, 240 updates on 192 windows "
                "(causal = prefix exactly: the reply never uses what the prompt read)",
                "this design, first run with three seeds: sort an 8-character prompt, two "
                "blocks of width 32 with 4 heads, reply fed back, 2000 updates (kept)",
                "pilot, validation part: the same with a 12-character prompt (causal 0.942, "
                "prefix 0.954 characters right; seeds overlap) - not kept",
                "pilot, validation part: the same with one head a block (causal 0.961, "
                "prefix 0.958; seeds overlap) - not kept",
            ],
        ),
    )


# ------------------------------------------------------------------ 4 multi-head-vs-single
CONTEXT = 16
HEAD_STEPS, HEAD_EVERY = 3000, 250
HEAD_COUNTS = (1, 8)
OFFSETS = 5                                 # 0 (itself) .. 4 back, then "5 or more back"
FROM = 5                                    # readers from this position on have all of them


def head_config(heads):
    return T.Config(vocab=len(R.vocabulary()), width=64, heads=heads, kv_heads=heads, ff=128,
                    layers=1, max_len=CONTEXT)


def head_heldout():
    """Held-out windows as next-character inputs and targets."""
    seq = _windows("test", CONTEXT + 1)
    return seq[:, :-1], seq[:, 1:]


def by_offset(attention):
    """Per head of ``attention`` (window, head, reader, read): the mean share of a reader's
    attention on itself, on the character 1, 2, 3 and 4 back, and on everything further
    back (the last column), over readers that have all of those: ``(head, 6)``."""
    n = attention.shape[-1]
    readers = np.arange(FROM, n)
    near = np.stack([attention[:, :, readers, readers - d].mean((0, 2)) for d in range(OFFSETS)], 1)
    return np.concatenate([near, 1.0 - near.sum(1, keepdims=True)], 1)


def even_spread():
    """:func:`by_offset` of a head that spreads its attention evenly over what it may read."""
    n = CONTEXT
    even = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]
    return by_offset(even[None, None])[0]


@lru_cache(maxsize=1)
def head_runs():
    """``{heads: [run per seed]}`` for one head and eight: parameters, training trace,
    held-out loss every 250 updates and at the end, and where each head looks."""
    xt, yt = head_heldout()

    def batch_of(text, rng):
        starts = rng.integers(0, len(text) - CONTEXT, BATCH)
        seq = text[starts[:, None] + np.arange(CONTEXT + 1)]
        return seq[:, :-1], seq[:, 1:], None

    out = {}
    for heads in HEAD_COUNTS:
        config = head_config(heads)
        runs = []
        for seed in SEEDS:
            run = _fit("heads-%d" % heads, config, batch_of, HEAD_STEPS, seed,
                       {"task": "next character", "context": CONTEXT,
                        "heldout": T.array_identity(xt)},
                       lambda P: {"heldout": T.loss(P, xt, yt, config)}, HEAD_EVERY)  # noqa: B023
            logits, records = T.forward(run["P"], xt, config, keep=True)
            run.update(config=config, attention=records["attention"][0],
                       heldout_loss=T.loss(run["P"], xt, yt, config),
                       accuracy=float((logits.argmax(-1) == yt).mean()))
            run["offsets"] = by_offset(run["attention"])
            runs.append(run)
        out[heads] = runs
    return out


def multi_head_vs_single():
    runs = head_runs()
    xt, yt = head_heldout()
    loss = np.array([[r["heldout_loss"] for r in runs[h]] for h in HEAD_COUNTS])
    curves = np.array([[r["probes"]["heldout"] for r in runs[h]] for h in HEAD_COUNTS])
    updates = np.arange(1, curves.shape[2] + 1) * HEAD_EVERY
    one, eight = runs[1][0]["offsets"], runs[8][0]["offsets"]
    labels = ["one head", "eight heads"]
    parameters = [counts(runs[h][0]["P"]) for h in HEAD_COUNTS]
    previous = [[r["offsets"][:, 1].tolist() for r in runs[h]] for h in HEAD_COUNTS]
    return Study(
        "One wide head or eight narrow heads",
        "Real text · Tiny Shakespeare · one-block NumPy transformers, three seeds each",
        (
            matrix(
                "Where each head looks, on held-out text",
                "Rows: the one head, then the eight heads (seed 0). Columns: share of "
                "attention on the reader itself, 1, 2, 3, 4 characters back, and further.",
                np.vstack([one, eight]),
                ("an even spread: " + ", ".join("%.0f%%" % (100 * v) for v in even_spread()),),
            ),
            curve(
                "Held-out loss as training goes on",
                "Next-character loss on 512 held-out windows; mean of three seeds.",
                curves.mean(1), labels, x=updates,
            ),
            bars(
                "Held-out loss of every seed",
                "Three one-head models, then three eight-head models.",
                loss.ravel(), ["%s, seed %d" % (name, s) for name in labels for s in SEEDS],
            ),
            bars(
                "Held-out next-character loss",
                "Mean of three seeds; both models have the same parameters, cut differently.",
                loss.mean(1), labels,
                ["%s: %.3f to %.3f" % (name, row.min(), row.max())
                 for name, row in zip(labels, loss)] + ["Parameters each: %d" % parameters[0]],
            ),
        ),
        _provenance(
            "next-character prediction",
            labels=labels, steps=HEAD_STEPS, context=CONTEXT, updates=updates.tolist(),
            configs=[asdict(head_config(h)) for h in HEAD_COUNTS], parameters=parameters,
            heldout_loss=loss.tolist(), heldout_loss_curves=curves.tolist(),
            accuracy=[[r["accuracy"] for r in runs[h]] for h in HEAD_COUNTS],
            previous_character_share=previous,
            self_share=[[r["offsets"][:, 0].tolist() for r in runs[h]] for h in HEAD_COUNTS],
            even_spread=even_spread().tolist(), guessing_evenly=float(np.log(len(R.vocabulary()))),
            heldout_arrays=[T.array_identity(a) for a in (xt, yt)],
            runs=[{"settings": dict(r["settings"])} for h in HEAD_COUNTS for r in runs[h]],
            designs_tried=[
                "first study: two blocks of width 32, 900 updates on 384 windows (both models "
                "memorised: held-out loss above even guessing)",
                "pilots, validation part, one seed: one block of width 32 as 1 head or 4 on "
                "12-character windows (the four heads differed less from each other); two "
                "blocks of width 64 as 1 head or 8 (twice the time); not kept",
                "this design: one block of width 64 as 1 head or 8, 16-character windows, "
                "3000 updates on fresh windows, three seeds (kept; where a head looks is "
                "measured by how far back, fixed before the test-part runs)",
            ],
        ),
    )


STUDIES = {
    "prefix-lm-attention": prefix_lm_attention,
    "multi-head-vs-single": multi_head_vs_single,
}

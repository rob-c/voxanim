"""Redesigned experiments for animations 7-12 of the "next 100" (see
:func:`voxanim.next_catalog.redesigned`).

The first experiments for ``sliding-window-attention`` and ``sparse-attention-patterns``
(:mod:`voxanim.next_attention`) trained every model on next-character prediction over
12-character windows. A small model predicts the next character almost entirely from the
last two or three, so how far a mask lets it reach made no measurable difference: the
runs overlapped, and deeper models only fitted worse in their 240 updates.

Both are replaced by the same real-text task, on which reach is the whole question:

**the delayed echo** - read a window of Tiny Shakespeare and, at every position, name the
character that stood ``d`` places earlier. The text and the targets are unmodified real
characters; windows for training come from the training part of the text and the
reported numbers from windows of the disjoint test part. A model whose mask (composed
over its layers) connects a position to the one ``d`` back can copy it; one whose mask
does not can only guess it from the characters it does see.

Every comparison is run on three seeds and reported as their mean with each seed kept.
"""

from __future__ import annotations

import json

import numpy as np

from . import realdata as R
from . import tinylm as T
from .cache import cached_npz
from .next_attention import _pattern_counts, sparse_mask
from .study import Panel, Study

SEEDS = (0, 1, 2)
TRAIN_WINDOWS, TEST_WINDOWS = 4096, 512
TRAIN_SEED, TEST_SEED = 37, 73
VERSION = 1


def echo_task(n, length, delay, split, seed):
    """``(x, y, w)``: ``n`` real text windows, and at each position the character ``delay``
    places earlier as its target (weight 0 where the window has no such character)."""
    x = R.text_sequences(n=n, length=length, split=split, seed=seed)
    y, w = np.zeros_like(x), np.zeros(x.shape)
    y[:, delay:] = x[:, :-delay]
    w[:, delay:] = 1.0
    return x, y, w


def answers(P, x, config, mask=None):
    """The character each position answers with."""
    return T.forward(P, x, config, mask=mask).argmax(-1)


def composed(mask, layers):
    """Which positions each position can have heard from after ``layers`` layers of
    ``mask``: the mask composed with itself (exact)."""
    reach = np.eye(len(mask), dtype=bool)
    for _ in range(layers):
        reach = (reach.astype(int) @ np.asarray(mask, int)) > 0
    return reach


def _provenance(task):
    return {
        "dataset": R.text_metadata("train"),
        "evaluation_dataset": R.text_metadata("test"),
        "task": task,
        "train_window_seed": TRAIN_SEED,
        "test_window_seed": TEST_SEED,
        "train_windows": TRAIN_WINDOWS,
        "test_windows": TEST_WINDOWS,
        "window_sampling": "without replacement within each source partition",
        "selection": "fixed settings; no validation-based model selection",
        "seeds": list(SEEDS),
        "source_sha256": R.source_hash("study_attention_b.py", "tinylm.py", "realdata.py"),
    }


def _fit(config, length, delay, seed, steps, batch, name, mask=None):
    """Train one model on the echo task; its trace, held-out answers and checkpoint hash."""
    x, y, w = echo_task(TRAIN_WINDOWS, length, delay, "train", TRAIN_SEED)
    xt, yt, wt = echo_task(TEST_WINDOWS, length, delay, "test", TEST_SEED)
    run = T.train(config, x, y, weights=w, steps=steps, seed=seed, batch=batch, mask=mask,
                  name=name, provenance={"task": "delayed echo", "delay": delay,
                                         "study": "study_attention_b", "version": VERSION})
    said = answers(run["P"], xt, config, mask)
    right = (said == yt) & (wt > 0)
    return {
        "trace": run["loss"],
        "right": right,                                   # per test window and position
        "accuracy": float(right.sum() / wt.sum()),
        "loss": T.loss(run["P"], xt, yt, config, wt, mask=mask),
        "said": said[0],                                  # the first held-out window
        "sha256": run["provenance"]["final_parameter_sha256"],
        "parameters": sum(v.size for v in run["P"].values()),
    }


# ---------------------------------------------------------------------- 7
WINDOW = 4
WINDOW_LENGTH = 20
WINDOW_DELAYS = (3, 4, 6, 7, 9, 10)     # each depth's furthest reach, and one place beyond it
WINDOW_LAYERS = (1, 2, 3)
WINDOW_STEPS, WINDOW_BATCH = 600, 16


def window_config(layers):
    return T.Config(vocab=len(R.vocabulary()), width=32, heads=4, kv_heads=4, ff=64,
                    mask="window", window=WINDOW, layers=layers, max_len=WINDOW_LENGTH)


def sliding_window():
    """Does a stack of window layers copy from exactly as far as its composed mask reaches?

    One model per depth (1, 2, 3 layers of a width-4 causal window), delay and seed. A
    window layer reads 3 places back, so ``L`` layers reach ``3 L``; the delays are each
    depth's limit (3, 6, 9) and the place just past it (4, 7, 10)."""
    settings = {
        "version": VERSION, "length": WINDOW_LENGTH, "window": WINDOW,
        "delays": WINDOW_DELAYS, "layers": WINDOW_LAYERS, "steps": WINDOW_STEPS,
        "batch": WINDOW_BATCH, "seeds": SEEDS, "provenance": _provenance("delayed echo"),
    }

    def build():
        shape = (len(WINDOW_LAYERS), len(WINDOW_DELAYS), len(SEEDS))
        accuracy, loss = np.zeros(shape), np.zeros(shape)
        traces = np.zeros(shape + (WINDOW_STEPS,))
        said = np.zeros(shape + (WINDOW_LENGTH,), int)
        params, hashes = [], {}
        for a, layers in enumerate(WINDOW_LAYERS):
            config = window_config(layers)
            for b, delay in enumerate(WINDOW_DELAYS):
                for c, seed in enumerate(SEEDS):
                    r = _fit(config, WINDOW_LENGTH, delay, seed, WINDOW_STEPS, WINDOW_BATCH,
                             "echo-window-%dL-d%d-s%d" % (layers, delay, seed))
                    accuracy[a, b, c], loss[a, b, c] = r["accuracy"], r["loss"]
                    traces[a, b, c], said[a, b, c] = r["trace"], r["said"]
                    hashes["%d layers, %d back, seed %d" % (layers, delay, seed)] = r["sha256"]
            params.append(r["parameters"])
        return dict(accuracy=accuracy, loss=loss, traces=traces, said=said,
                    parameters=np.array(params), checkpoints=np.array(json.dumps(hashes)))

    d = cached_npz("study-sliding-window-echo", settings, build)
    accuracy, loss = d["accuracy"], d["loss"]
    band = T.attention_mask(12, window_config(1))
    covers = np.array([[delay <= layers * (WINDOW - 1) for delay in WINDOW_DELAYS]
                       for layers in WINDOW_LAYERS])
    # the same statement from the trained mask itself, composed (exact)
    trained = T.attention_mask(WINDOW_LENGTH, window_config(1))
    assert all(composed(trained, layers)[WINDOW_LENGTH - 1, WINDOW_LENGTH - 1 - delay]
               == covers[a, b] for a, layers in enumerate(WINDOW_LAYERS)
               for b, delay in enumerate(WINDOW_DELAYS))
    names = ["%d layer%s" % (n, "" if n == 1 else "s") for n in WINDOW_LAYERS]
    shown = WINDOW_DELAYS.index(9)                         # the task whose traces are drawn
    example = R.text_sequences(n=TEST_WINDOWS, length=WINDOW_LENGTH, split="test",
                               seed=TEST_SEED)[0]
    return Study(
        "Sliding-window attention",
        "Window-4 models of 1-3 layers asked for the character d places back in real text",
        (
            Panel("Reach after three local layers",
                  "A width-four mask reaches at most nine previous positions after three "
                  "layers (exact mask composition).", "matrix",
                  {"values": composed(band, 3).astype(float)}),
            Panel("Learning to name the character 9 back",
                  "Minibatch loss before each update; mean of the three seeds, and each seed.",
                  "curve",
                  {"x": np.arange(1, WINDOW_STEPS + 1), "ys": d["traces"][:, shown].mean(1),
                   "labels": names, "each": d["traces"][:, shown], "delay": 9}),
            Panel("Held-out accuracy against the distance asked for",
                  "Share of held-out positions answered with the right character; mean of "
                  "three seeds, with the lowest and highest seed.", "curve",
                  {"x": np.array(WINDOW_DELAYS), "ys": accuracy.mean(2), "labels": names,
                   "min": accuracy.min(2), "max": accuracy.max(2), "each": accuracy}),
            Panel("Held-out loss, depth by distance",
                  "Mean held-out loss in nats; rows are 1, 2, 3 layers, columns the distance.",
                  "matrix", {"values": loss.mean(2), "each": loss}),
        ),
        {
            **_provenance("delayed echo: at every position, name the character d places "
                          "earlier in the same real-text window"),
            "design": "depth x distance x seed; one model each; the control is the distance "
                      "one place beyond each depth's reach",
            "length": WINDOW_LENGTH, "window": WINDOW, "delays": list(WINDOW_DELAYS),
            "layers": list(WINDOW_LAYERS), "steps": WINDOW_STEPS, "batch": WINDOW_BATCH,
            "width": 32, "parameters": [int(v) for v in d["parameters"]],
            "reach_back": [n * (WINDOW - 1) for n in WINDOW_LAYERS],
            "reach_covers": covers.tolist(),
            "heldout_accuracy": accuracy.tolist(),          # [depth][distance][seed]
            "heldout_loss": loss.tolist(),
            "example": {"window": [int(v) for v in example],
                        "seed": SEEDS[0],
                        "said": d["said"][:, :, 0].tolist(),  # [depth][distance][position]
                        "note": "the first held-out window and the first seed's models, "
                                "fixed before any result was seen"},
            "checkpoints": json.loads(str(d["checkpoints"])),
            "designs_tried": [
                "original: next-character prediction, 12-character windows, 240 updates, one "
                "seed: 2.95 / 3.08 / 3.16 nats for 1/2/3 layers (depth did not help)",
                "pilot, one seed, 20-character echo, width 32, batch 32: 1 layer on 3 and 6 "
                "back, 2 layers on 6 back, 3 layers on 9 back, to see whether the task is "
                "learnable and how many updates the 3-layer model needs (about 400-600)",
                "pilot, 3 seeds, 3 layers on 9 back at batch 16: 99% by 600 updates on each; "
                "this fixed 600 updates and batch 16 for every model",
                "final (this study): the full depth x distance grid at those settings, run once",
            ],
        },
    )


# ---------------------------------------------------------------------- 8
KINDS = ("dense", "strided", "block", "global", "random")
SPARSE_LENGTH, SPARSE_DELAY, SPARSE_LAYERS = 32, 10, 2
SPARSE_STEPS, SPARSE_BATCH = 1000, 16
COUNT_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def sparse_config():
    return T.Config(vocab=len(R.vocabulary()), width=48, heads=4, kv_heads=4, ff=96,
                    layers=SPARSE_LAYERS, max_len=SPARSE_LENGTH)


def sparse_patterns():
    """What does each sparse pattern give up for the pairs it saves?

    Five causal masks over 32 characters (:func:`voxanim.next_attention.sparse_mask`, local
    window and stride 4), each in a two-layer model asked for the character 10 back: further
    than two local hops (6). Whether two hops of a mask connect a position to the one 10
    back is exact; whether a model learns to use the connection is measured."""
    settings = {
        "version": VERSION, "length": SPARSE_LENGTH, "delay": SPARSE_DELAY, "kinds": KINDS,
        "layers": SPARSE_LAYERS, "steps": SPARSE_STEPS, "batch": SPARSE_BATCH, "seeds": SEEDS,
        "provenance": _provenance("delayed echo"),
        "masks": R.source_hash("next_attention.py"),
    }
    n, delay = SPARSE_LENGTH, SPARSE_DELAY
    masks = [sparse_mask(n, kind) for kind in KINDS]
    asked = np.arange(delay, n)                             # positions that have a target
    reach = np.array([composed(m, SPARSE_LAYERS)[asked, asked - delay] for m in masks])

    def build():
        config = sparse_config()
        shape = (len(KINDS), len(SEEDS))
        accuracy, loss = np.zeros(shape), np.zeros(shape)
        within, beyond = np.zeros(shape), np.zeros(shape)   # right answers, by reachability
        traces = np.zeros(shape + (SPARSE_STEPS,))
        said = np.zeros(shape + (n,), int)
        hashes = {}
        for a, (kind, mask) in enumerate(zip(KINDS, masks)):
            for c, seed in enumerate(SEEDS):
                r = _fit(config, n, delay, seed, SPARSE_STEPS, SPARSE_BATCH,
                         "echo-sparse-%s-s%d" % (kind, seed), mask=mask)
                accuracy[a, c], loss[a, c] = r["accuracy"], r["loss"]
                traces[a, c], said[a, c] = r["trace"], r["said"]
                right = r["right"][:, asked]
                within[a, c] = right[:, reach[a]].sum()
                beyond[a, c] = right[:, ~reach[a]].sum()
                hashes["%s, seed %d" % (kind, seed)] = r["sha256"]
        return dict(accuracy=accuracy, loss=loss, traces=traces, said=said, within=within,
                    beyond=beyond, parameters=np.array(r["parameters"]),
                    checkpoints=np.array(json.dumps(hashes)))

    d = cached_npz("study-sparse-pattern-echo", settings, build)

    def build_counts():
        counts = [[_pattern_counts(int(m), kind) for m in COUNT_LENGTHS] for kind in KINDS]
        return dict(n=np.array(COUNT_LENGTHS), counts=np.array(counts))

    big = cached_npz("study-sparse-pattern-counts",
                     {"version": VERSION, "source": R.source_hash("next_attention.py")},
                     build_counts)
    accuracy = d["accuracy"]
    pairs = [int(m.sum()) for m in masks]
    n_within = reach.sum(1) * TEST_WINDOWS                  # held-out positions of each kind
    n_beyond = (~reach).sum(1) * TEST_WINDOWS
    example = R.text_sequences(n=TEST_WINDOWS, length=n, split="test", seed=TEST_SEED)[0]
    return Study(
        "Sparse attention patterns",
        "Five causal masks over 32 characters, each asked for the character 10 back",
        (
            Panel("Five wiring patterns",
                  "Dense, strided, block-local, global-token and random-local masks over 32 "
                  "characters, stacked: the masks the models were trained under.", "matrix",
                  {"values": np.concatenate(masks, axis=0).astype(float)}),
            Panel("Train under each mask",
                  "Minibatch loss before each update; mean of the three seeds, and each seed.",
                  "curve",
                  {"x": np.arange(1, SPARSE_STEPS + 1), "ys": d["traces"].mean(1),
                   "labels": list(KINDS), "each": d["traces"]}),
            Panel("Held-out accuracy at the character 10 back",
                  "Share of held-out positions answered with the right character; mean of "
                  "three seeds, with the lowest and highest seed.", "bars",
                  {"values": accuracy.mean(1), "labels": list(KINDS), "min": accuracy.min(1),
                   "max": accuracy.max(1), "each": accuracy, "pairs": np.array(pairs),
                   "reachable": reach.mean(1)}),
            Panel("Count actual allowed pairs",
                  "Fixed-window block, global and random patterns remain linear in sequence "
                  "length (exact counts).", "curve",
                  {"x": big["n"], "ys": big["counts"], "labels": list(KINDS)}),
        ),
        {
            **_provenance("delayed echo: at every position, name the character 10 places "
                          "earlier in the same real-text window"),
            "design": "pattern x seed; one two-layer model each; the control is exact: the "
                      "positions two hops of the mask can and cannot connect to their target",
            "kinds": list(KINDS), "length": n, "delay": delay, "layers": SPARSE_LAYERS,
            "steps": SPARSE_STEPS, "batch": SPARSE_BATCH, "width": 48,
            "parameters": int(d["parameters"]),
            "pairs": pairs,
            "asked_positions": [int(v) for v in asked],
            "reachable": reach.astype(int).tolist(),         # [pattern][asked position]
            "reachable_share": [float(v) for v in reach.mean(1)],
            "heldout_accuracy": accuracy.tolist(),           # [pattern][seed]
            "heldout_loss": d["loss"].tolist(),
            # right answers at the positions two hops connect / do not connect, per seed
            "right_where_reachable": d["within"].tolist(),
            "right_where_unreachable": d["beyond"].tolist(),
            "positions_reachable": [int(v) for v in n_within],
            "positions_unreachable": [int(v) for v in n_beyond],
            "example": {"window": [int(v) for v in example], "seed": SEEDS[0],
                        "said": d["said"][:, 0].tolist(),    # [pattern][position]
                        "note": "the first held-out window and the first seed's models, "
                                "fixed before any result was seen"},
            "checkpoints": json.loads(str(d["checkpoints"])),
            "designs_tried": [
                "original: next-character prediction, 12-character windows, 200 updates, one "
                "seed: held-out 2.74-2.95 nats for all five patterns (no visible difference)",
                "pilot, one seed, this task at width 32, batch 32, 600 updates: dense 100%, "
                "strided 66% and still improving, block 15%, global 20%, random 63%",
                "pilot, strided only, one seed, 2000 updates at width 32 and 48: 95% and 99%; "
                "at width 48, 93% by 1000 updates. This fixed width 48 and, for the build "
                "time, 1000 updates for every pattern",
                "final (this study): all five patterns, three seeds, at those settings, run "
                "once",
            ],
        },
    )


STUDIES = {
    "sliding-window-attention": sliding_window,
    "sparse-attention-patterns": sparse_patterns,
}

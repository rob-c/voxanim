"""Redesigned experiments of the transformer-block section (animations 13-20).

The first experiments (:mod:`voxanim.next_attention`) trained each variant once, for 100
to 260 updates on 192 twelve-character windows. That could not show what these five
animations are about:

* ``pre-ln-vs-post-ln``: 100 updates end before a deep model has left the loss of guessing
  by letter frequency, so the two placements looked the same. Here every arm gets 800
  updates at one fixed rate with no warm-up, post-LN is also run with a warm-up (the
  control), and each is repeated over three seeds at 2 and at 12 blocks.
* ``transformer-depth-vs-width``: the wide models memorised their 192 windows. Here the
  training text is 32,000 windows, each drawn about once (512,000 characters, less than
  one pass over the training split), the four shapes hold the same number of weights to
  0.3%, and each is trained three times.
* ``rmsnorm-vs-layernorm``, ``swiglu-mlp``, ``parallel-block``: one seed each, with
  differences of 0.007 to 0.13 nats. Here each variant is trained eight times; the
  per-seed differences say whether a gap is larger than the seed-to-seed spread.

Everything is real text (Tiny Shakespeare): windows of the training split to learn from,
1,024 windows of the disjoint test split to score on. A seed sets a model's initial
weights and the order of its minibatches; variants compared share seeds, windows and
every other setting. All settings were fixed before the comparisons were looked at,
except where ``metadata["designs_tried"]`` says otherwise.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np

from . import realdata as R
from . import tinylm as T
from .cache import cached_npz
from .study import Panel, Study

LR = 0.003                   # tinylm's default rate, used for every run here
TRAIN_SEED, TEST_SEED, TEST_WINDOWS = 37, 73, 1024
BLOCK = 20                   # updates averaged into one point of a drawn training trace
CONTEXT_WINDOWS = 8192       # test windows the loss by position (study 20) is averaged over


def base_config(max_len=12):
    """The one-block model the three variant studies change one thing in."""
    return T.Config(vocab=len(R.vocabulary()), width=16, heads=4, kv_heads=4, ff=32,
                    max_len=max_len)


# name: (configs, labels, warm-up updates per config, seeds, updates, batch, window, windows)
def designs(name):
    c = base_config()
    if name == "rmsnorm-vs-layernorm":
        return ([c, replace(c, norm="rms")], ["LayerNorm", "RMSNorm"], (0, 0), range(8), 2000,
                32, 12, 32768)
    if name == "swiglu-mlp":
        return ([replace(c, ff=48), replace(c, ff=32, activation="swiglu")], ["GELU", "SwiGLU"],
                (0, 0), range(8), 2000, 32, 12, 32768)
    if name == "parallel-block":
        return ([c, replace(c, parallel=True)], ["sequential", "parallel"], (0, 0), range(8),
                2000, 32, 12, 32768)
    if name == "pre-ln-vs-post-ln":
        configs = [replace(c, layers=depth, placement=place) for depth in LN_DEPTHS
                   for place in ("pre", "post", "post")]
        labels = ["%s, %d blocks" % (arm, depth) for depth in LN_DEPTHS for arm in LN_ARMS]
        return configs, labels, (0, 0, LN_WARMUP) * 2, range(3), 800, 24, 12, 16384
    if name == "transformer-depth-vs-width":
        c = base_config(max_len=16)
        configs = [replace(c, layers=n, width=d, ff=ff) for n, d, ff in SHAPES]
        return configs, ["%d x %d" % (n, d) for n, d, _ff in SHAPES], (0,) * 4, range(3), 1000, \
            32, 16, 32000
    raise KeyError(name)


LN_DEPTHS = (2, 12)
LN_ARMS = ("pre-LN", "post-LN", "post-LN + warm-up")
LN_WARMUP = 100
# blocks, width, MLP hidden units: 62,848 / 62,880 / 62,912 / 62,736 learned numbers
SHAPES = ((1, 64, 286), (2, 48, 192), (4, 32, 161), (8, 24, 104))


def counts(P):
    return int(sum(v.size for v in P.values()))


def token_nll(P, x, y, config):
    """Loss of every character of the windows ``x``: ``(windows, length)`` nats."""
    out = []
    for i in range(0, len(x), 256):
        z = T.forward(P, x[i:i + 256], config)
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        yy = y[i:i + 256]
        out.append(-logp[np.arange(len(yy))[:, None], np.arange(yy.shape[1]), yy])
    return np.concatenate(out)


@cache
def experiment(name):
    """Train (or load) every run of the study ``name``. A dict: ``configs``, ``labels``,
    ``seeds``, ``steps``, ``runs[variant][seed]`` (``tinylm.train`` records), ``test``
    (``x, y``), ``nll`` ``(variants, seeds, length)`` the held-out loss by position in the
    window, ``heldout`` ``(variants, seeds)`` and ``traces`` ``(variants, seeds, updates)``."""
    configs, labels, warmups, seeds, steps, batch, length, windows = designs(name)
    x, y, _w = R.language_data(n=windows, length=length, split="train", seed=TRAIN_SEED)
    xt, yt, _w = R.language_data(n=TEST_WINDOWS, length=length, split="test", seed=TEST_SEED)
    provenance = {
        "dataset": R.text_metadata("train"),
        "evaluation_dataset": R.text_metadata("test"),
        "task": "next-character prediction",
        "selection": "fixed settings; no validation-based model selection",
        "train_window_seed": TRAIN_SEED,
        "test_window_seed": TEST_SEED,
        "window_sampling": "without replacement within each source partition",
        "evaluation_arrays": [T.array_identity(a) for a in (xt, yt)],
    }
    runs = [[T.train(c, x, y, steps=steps, lr=LR, seed=seed, warmup=warm, batch=batch,
                     name="block-%s-%d" % (name, i), provenance=provenance)
             for seed in seeds] for i, (c, warm) in enumerate(zip(configs, warmups))]
    nll = np.array([[token_nll(r["P"], xt, yt, c).mean(0) for r in rs]
                    for c, rs in zip(configs, runs)])
    return dict(
        name=name, configs=configs, labels=labels, warmups=list(warmups), seeds=list(seeds),
        steps=steps, batch=batch, length=length, windows=windows, runs=runs, train=(x, y),
        test=(xt, yt), nll=nll, heldout=nll.mean(-1),
        traces=np.array([[r["loss"] for r in rs] for rs in runs]),
        params=[counts(rs[0]["P"]) for rs in runs],
    )


def smooth(traces):
    """``(..., updates)`` minibatch losses as means of :data:`BLOCK` consecutive updates;
    with the update number each mean ends on."""
    traces = np.asarray(traces, float)
    n = traces.shape[-1] // BLOCK
    return (np.arange(1, n + 1) * BLOCK,
            traces[..., :n * BLOCK].reshape(traces.shape[:-1] + (n, BLOCK)).mean(-1))


def curve(title, caption, ys, labels, notes=(), x=None):
    ys = np.atleast_2d(ys)
    return Panel(title, caption, "curve",
                 {"x": np.arange(ys.shape[1]) if x is None else np.asarray(x), "ys": ys,
                  "labels": list(labels)}, tuple(notes))


def bars(title, caption, values, labels, notes=()):
    return Panel(title, caption, "bars", {"values": np.asarray(values), "labels": list(labels)},
                 tuple(notes))


def matrix(title, caption, values, notes=()):
    return Panel(title, caption, "matrix", {"values": np.asarray(values)}, tuple(notes))


def metadata(e, evidence, **extra):
    held = e["heldout"]
    return {
        "evidence_kind": "trained_model_measurement",
        "panel_evidence": evidence,
        "dataset": R.text_metadata("train"),
        "evaluation_dataset": R.text_metadata("test"),
        "task": "next-character prediction",
        "labels": e["labels"],
        "seeds": e["seeds"],
        "seed_meaning": "initial weights and minibatch order; data and settings are shared",
        "steps": e["steps"],
        "batch": e["batch"],
        "window_length": e["length"],
        "training_windows": e["windows"],
        "heldout_windows": TEST_WINDOWS,
        "learning_rate": LR,
        "warmup_updates": e["warmups"],
        "parameters": e["params"],
        "heldout_loss": held.mean(1).tolist(),
        "heldout_loss_by_seed": held.tolist(),
        "heldout_loss_min": held.min(1).tolist(),
        "heldout_loss_max": held.max(1).tolist(),
        "source_sha256": R.source_hash("study_block.py", "tinylm.py", "realdata.py"),
        "runs": [r["provenance"] for rs in e["runs"] for r in rs],
        "limitations": [
            "Small models and short budgets on one corpus; no universal architecture ranking",
            "Seeds vary initial weights and minibatch order, not the text",
        ],
        **extra,
    }


def spread(values):
    return "%.3f (%.3f to %.3f)" % (np.mean(values), np.min(values), np.max(values))


# ---------------------------------------------------------------------- 14, 15, 18
def paired(name, title, question):
    """Two variants of one model, eight seeds each: is the gap larger than the spread?"""
    e = experiment(name)
    a, b = e["labels"]
    held = e["heldout"]
    diff = held[1] - held[0]                              # per seed: second minus first
    x, traces = smooth(e["traces"])
    return Study(
        title,
        "Real text · Tiny Shakespeare · %d seeds of each variant, %d updates" % (
            len(e["seeds"]), e["steps"]),
        (
            matrix(
                "Held-out loss of every run",
                "Rows: %s, %s. Columns: the %d seeds. Nats per character on %d held-out "
                "windows." % (a, b, len(e["seeds"]), TEST_WINDOWS),
                held,
            ),
            curve(
                "Training, averaged over the seeds",
                "Each point is the mean minibatch loss of %d consecutive updates and all "
                "seeds." % BLOCK,
                traces.mean(1), e["labels"], x=x,
            ),
            bars(
                "Mean held-out loss",
                question,
                held.mean(1), e["labels"],
                ("%s %s" % (a, spread(held[0])), "%s %s" % (b, spread(held[1])),
                 "nats per character, lower is better; mean (lowest to highest seed)"),
            ),
            bars(
                "Count every learned parameter",
                "The comparison includes embeddings and normalization parameters.",
                e["params"], e["labels"],
            ),
        ),
        metadata(
            e, ["trained_model_measurement"] * 3 + ["analytic"],
            paired_difference=diff.tolist(),
            paired_difference_mean=float(diff.mean()),
            paired_difference_sd=float(diff.std(ddof=1)),
            seeds_second_is_better=int((diff < 0).sum()),
            seed_sd=held.std(1, ddof=1).tolist(),
            first_study="one seed of each, 240 updates on 192 windows (next_attention)",
        ),
    )


def rmsnorm_vs_layernorm():
    return paired("rmsnorm-vs-layernorm", "RMSNorm and LayerNorm",
                  "Dropping the mean subtraction: the same model trained with each norm.")


def swiglu_mlp():
    return paired("swiglu-mlp", "A gated transformer MLP",
                  "48 GELU units against 32 gated units: the same number of MLP weights.")


def parallel_block():
    return paired("parallel-block", "Sequential and parallel blocks",
                  "The MLP reads the stream after attention's write, or beside it.")


# ---------------------------------------------------------------------- 13
@cache
def initial_gradients():
    """Size of the loss gradient at each block's query projection before any update, in
    the 12-block pre-LN and post-LN models: ``(2, seeds, 12)``, on the first minibatch's
    worth of training windows."""
    e = experiment("pre-ln-vs-post-ln")
    x, y = (a[:e["batch"]] for a in e["train"])
    out = []
    for cfg in e["configs"][3:5]:
        rows = []
        for seed in e["seeds"]:
            _loss, grad = T.loss_grads(T.init(cfg, seed), x, y, cfg)
            rows.append([np.linalg.norm(grad["l%d.Q" % i]) for i in range(cfg.layers)])
        out.append(rows)
    return np.array(out)


def pre_ln_vs_post_ln():
    e = experiment("pre-ln-vs-post-ln")
    held = e["heldout"]                                   # (2 depths x 3 arms, seeds)
    x, traces = smooth(e["traces"])
    grads = initial_gradients()
    deep = LN_DEPTHS[1]
    notes = tuple("%s: %s" % (label, spread(h)) for label, h in zip(e["labels"], held))
    return Study(
        "Pre-LN and post-LN with depth",
        "Real text · %d and %d blocks · one fixed rate, with and without warm-up · %d seeds" % (
            LN_DEPTHS + (len(e["seeds"]),)),
        (
            curve(
                "Gradients through %d blocks" % deep,
                "Size of the gradient at each block's query projection before the first "
                "update; mean of %d initialisations." % len(e["seeds"]),
                grads.mean(1), ["pre-LN", "post-LN"], x=np.arange(1, deep + 1),
            ),
            curve(
                "Training %d blocks" % deep,
                "Mean over seeds of %d-update means. Rate %g from the first update, or "
                "raised linearly over %d updates (warm-up)." % (BLOCK, LR, LN_WARMUP),
                traces[3:].mean(1), LN_ARMS, x=x,
            ),
            curve(
                "Training %d blocks" % LN_DEPTHS[0],
                "The same three arms on a shallow model.",
                traces[:3].mean(1), LN_ARMS, x=x,
            ),
            bars(
                "Held-out loss at both depths",
                "Mean over %d seeds on the same %d disjoint test-text windows." % (
                    len(e["seeds"]), TEST_WINDOWS),
                held.mean(1), e["labels"], notes,
            ),
        ),
        metadata(
            e, ["untrained_model_measurement"] + ["trained_model_measurement"] * 3,
            depths=list(LN_DEPTHS), arms=list(LN_ARMS),
            initial_gradients_by_seed=grads.tolist(),
            designs_tried=[
                "first study: 4/12/24 blocks of width 8, 100 updates, one seed, no warm-up arm "
                "(no difference between placements)",
                "pilot, one seed, 12 blocks of width 16, 400 updates, rates 0.001 / 0.003 / "
                "0.01, arms pre, post, post + 80 warm-up updates: held-out 2.73 / 2.83 / 2.97 "
                "at 0.001; 2.53 / 3.33 / 2.85 at 0.003; 2.45 / 3.33 / 3.33 at 0.01. The rate "
                "0.003 (the library default) was kept; depths, 800 updates, the 100-update "
                "warm-up and three seeds were then fixed before this run",
            ],
        ),
    )


# ---------------------------------------------------------------------- 20
def depth_vs_width():
    e = experiment("transformer-depth-vs-width")
    held = e["heldout"]
    x, traces = smooth(e["traces"])
    last = e["traces"][:, :, -100:].mean(-1)              # (shapes, seeds): the fit at the end
    xc, yc, _w = R.language_data(n=CONTEXT_WINDOWS, length=e["length"], split="test",
                                 seed=TEST_SEED + 1)

    def by_context():
        return {"nll": np.array([[token_nll(r["P"], xc, yc, c).mean(0) for r in rs]
                                 for c, rs in zip(e["configs"], e["runs"])])}

    context = cached_npz("block-depth-width-context", {
        "version": 1, "windows": [T.array_identity(a) for a in (xc, yc)],
        "checkpoints": [r["provenance"]["final_parameter_sha256"] for rs in e["runs"] for r in rs],
    }, by_context)["nll"]                                 # (shapes, seeds, length)
    by_position = context.mean(1)
    notes = tuple("%s: %s" % (label, spread(h)) for label, h in zip(e["labels"], held))
    return Study(
        "Transformer depth and width",
        "Real text · equal weights as 1, 2, 4 and 8 blocks · %d seeds, %d updates" % (
            len(e["seeds"]), e["steps"]),
        (
            bars(
                "The same weights, four shapes",
                "Every learned number is counted; the four totals agree to 0.3%.",
                e["params"], e["labels"],
            ),
            curve(
                "Training on text never repeated",
                "Mean over seeds of %d-update means; %s windows are drawn from, each about "
                "once: less than one pass over the training text." % (
                    BLOCK, "{:,}".format(e["windows"])),
                traces.mean(1), e["labels"], x=x,
            ),
            bars(
                "Held-out loss of each shape",
                "Mean over %d seeds on %d windows of disjoint test text." % (
                    len(e["seeds"]), TEST_WINDOWS),
                held.mean(1), e["labels"], notes,
            ),
            curve(
                "Held-out loss by characters of context",
                "Loss of predicting the next character after seeing 1..%d characters, mean "
                "over seeds, on a larger sample of the test text (%s windows) so that the "
                "positions can be told apart." % (e["length"], f"{CONTEXT_WINDOWS:,}"),
                by_position, e["labels"], x=np.arange(1, e["length"] + 1),
            ),
        ),
        metadata(
            e, ["analytic"] + ["trained_model_measurement"] * 3,
            shapes=[dict(blocks=n, width=d, mlp_units=ff) for n, d, ff in SHAPES],
            final_training_loss=last.mean(1).tolist(),
            final_training_loss_by_seed=last.tolist(),
            final_training_loss_meaning="mean minibatch loss of the last 100 updates",
            heldout_loss_by_position=by_position.tolist(),
            heldout_loss_by_position_by_seed=context.tolist(),
            heldout_loss_by_position_windows=CONTEXT_WINDOWS,
            heldout_loss_by_position_window_seed=TEST_SEED + 1,
            uniform_guess=float(np.log(len(R.vocabulary()))),
            designs_tried=[
                "first study: 2/4/8 blocks, 260 updates on 192 windows, one seed, unequal "
                "weights (the wide models memorised)",
            ],
            budget_caveat="All four shapes are still improving at the last update and the "
            "deeper ones are behind throughout; a longer budget could change the order.",
            not_measured="No two-hop composition task: the data rule excludes synthetic "
            "tasks, and at this budget the models learn short-range letter statistics, which "
            "one block can represent. The study reports what depth buys on real text at "
            "equal weights and equal updates, not a composition result.",
        ),
    )


STUDIES = {
    "pre-ln-vs-post-ln": pre_ln_vs_post_ln,
    "rmsnorm-vs-layernorm": rmsnorm_vs_layernorm,
    "swiglu-mlp": swiglu_mlp,
    "parallel-block": parallel_block,
    "transformer-depth-vs-width": depth_vs_width,
}

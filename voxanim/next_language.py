"""Measured studies for the language, alignment and interpretability review set.

Every figure is built from the arrays produced here. Cached studies record their
seed, implementation version, and dataset; unexpected outcomes are displayed as
measured rather than replaced by the proposed story in TODO.md.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict
from functools import lru_cache

import numpy as np

from . import gpt2, gpt2_hooks, realdata, toycorpus
from .cache import cached_npz
from .optim import Adam

VERSION = 8
SEED = 23


@lru_cache(maxsize=1)
def _corpus():
    vocabulary = realdata.vocabulary()
    return dict(
        train=realdata.text("train"),
        test=realdata.text("test"),
        validation=realdata.text("validation"),
        alphabet=vocabulary,
        lookup={ch: i for i, ch in enumerate(vocabulary)},
        vocab=len(vocabulary),
        sha256=realdata.TEXT_SHA256,
    )


def _text_meta(**extra):
    return dict(
        dataset="Tiny Shakespeare / char-rnn",
        dataset_sha256=realdata.TEXT_SHA256,
        dataset_provenance={k: realdata.text_metadata(k) for k in ("train", "validation", "test")},
        **extra,
    )


def _encode_chars(text):
    return realdata.encode(text)


def _text_windows(n=192, length=16, split="train", seed=SEED, text=None):
    if text is None:
        return realdata.text_sequences(n, length + 1, split, seed)
    values = realdata.encode(text)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(values) - length, n, replace=False)
    return values[starts[:, None] + np.arange(length + 1)]


def _domain_windows(domain, split="train", n=192, length=16):
    """Two source regions, each with its own contiguous 80/20 train/test split."""
    text = _corpus()["train"]
    half = len(text) // 2
    region = text[domain * half : (domain + 1) * half]
    cut = int(len(region) * 0.8)
    chosen = region[:cut] if split == "train" else region[cut:]
    return _text_windows(n, length, seed=SEED + 20 * domain + (split != "train"), text=chosen)


def _save_weights(name, params, history=None, dataset=None, **settings):
    return realdata.save_evidence(
        "language-" + name,
        params,
        history or {},
        dataset if dataset is not None else realdata.text_metadata("train"),
        dict(version=VERSION, seed=SEED, **settings),
        source_files=("next_language.py", "tinylm.py"),
    )


GPT2_NAMES = {
    "perplexity-explained",
    "beam-search",
    "speculative-decoding",
    "repetition-and-penalties",
    "constrained-decoding",
    "contrastive-decoding",
    "knowledge-editing",
    "sparse-autoencoder-features",
    "linear-probes",
    "activation-patching",
    "induction-heads",
    "head-ablation",
}


def _panel(title, caption, kind="curve", notes=(), **data):
    from .study import Panel

    return Panel(title=title, caption=caption, kind=kind, data=data, notes=tuple(notes))


def _curve(title, caption, ys, labels, x=None, notes=()):
    ys = np.atleast_2d(ys)
    return _panel(
        title,
        caption,
        "curve",
        notes,
        x=np.arange(ys.shape[1]) if x is None else np.asarray(x),
        ys=ys,
        labels=list(labels),
    )


def _bars(title, caption, values, labels, notes=()):
    return _panel(title, caption, "bars", notes, values=np.asarray(values), labels=list(labels))


def _matrix(title, caption, values, notes=()):
    return _panel(title, caption, "matrix", notes, values=np.asarray(values))


def _study(title, subtitle, panels, **metadata):
    from .study import Study

    return Study(
        title=title,
        subtitle=subtitle,
        panels=tuple(panels),
        metadata=dict(seed=SEED, study_version=VERSION, **metadata),
    )


def _cache(name, create):
    def build():
        study = create()
        output, descriptions = {}, []
        for i, panel in enumerate(study.panels):
            desc = asdict(panel)
            desc["data"] = {}
            for key, value in panel.data.items():
                if isinstance(value, np.ndarray):
                    output["p%d_%s" % (i, key)] = value
                    desc["data"][key] = {"array": "p%d_%s" % (i, key)}
                else:
                    desc["data"][key] = value
            descriptions.append(desc)
        dataset = study.metadata.get("dataset_provenance")
        if dataset is None:
            dataset = {
                "source": study.metadata.get("source", study.subtitle),
                "dataset": study.metadata.get(
                    "dataset", "pretrained inference or exact computation"
                ),
            }
        study.metadata["measurement_evidence"] = realdata.save_evidence(
            "language-study-" + name,
            {},
            output,
            dataset,
            dict(seed=SEED, version=VERSION, study=name, **study.metadata.get("settings", {})),
            source_files=("next_language.py", "tinylm.py", "gpt2_hooks.py"),
        )
        output["description"] = np.array(
            json.dumps(
                dict(
                    title=study.title,
                    subtitle=study.subtitle,
                    panels=descriptions,
                    metadata=study.metadata,
                )
            )
        )
        return output

    settings = dict(
        version=VERSION,
        seed=SEED,
        study=name,
        gpt2_revision=gpt2.REVISION,
        dataset_sha256=_corpus()["sha256"],
        source_sha256=realdata.source_hash("next_language.py", "realdata.py", "tinylm.py"),
    )
    if name in {
        "reward-hacking",
        "grokking",
        "saliency-and-integrated-gradients",
        "packing-and-padding",
        "dpo",
    }:
        settings["revision"] = 1
    arrays = cached_npz("next-language-" + name, settings, build)
    from .study import Panel, Study

    description = json.loads(str(arrays["description"]))
    for panel in description["panels"]:
        panel["notes"] = tuple(panel["notes"])
        for key, value in panel["data"].items():
            if isinstance(value, dict) and "array" in value:
                panel["data"][key] = arrays[value["array"]]
    description["panels"] = tuple(Panel(**p) for p in description["panels"])
    return Study(**description)


def nll(logits, labels):
    p = gpt2.softmax(logits)
    return -np.log(
        np.maximum(np.take_along_axis(p, np.asarray(labels)[..., None], -1)[..., 0], 1e-300)
    )


@lru_cache(maxsize=1)
def _tok():
    return gpt2.Tokenizer()


def _forward(text, **kw):
    return gpt2_hooks.forward(_tok().encode(text), **kw)


def _perplexity():
    from collections import Counter

    train_ids = _tok().encode(_corpus()["train"][:30000])
    source = _corpus()["test"]
    excerpts = [source[i : i + 100] for i in range(0, 800, 100)]
    counts = np.bincount(train_ids, minlength=gpt2.N_VOCAB).astype(float)
    unigram = (counts + 0.1) / (counts.sum() + 0.1 * gpt2.N_VOCAB)
    pairs = Counter(itertools.pairwise(train_ids))
    totals = Counter(train_ids[:-1])
    losses = []
    uni = []
    bi = []
    first_ids = None
    for text in excerpts:
        ids = _tok().encode(text)
        if first_ids is None:
            first_ids = ids
        losses.append(nll(gpt2_hooks.forward(ids)["logits"][:-1], ids[1:]))
        uni.extend(-np.log(unigram[ids[1:]]))
        bi.extend(
            -np.log((pairs[(a, b)] + 0.1) / (totals[a] + 0.1 * gpt2.N_VOCAB))
            for a, b in itertools.pairwise(ids)
        )
    all_loss = np.concatenate(losses)
    ppl = np.exp([np.mean(uni), np.mean(bi), all_loss.mean()])
    return _study(
        "Perplexity on collected text",
        "TinyShakespeare · identical GPT-2 token targets for all models",
        [
            _bars(
                "Surprise at actual source tokens",
                "The first eight scored tokens of one held-out excerpt; values are negative log"
                " probabilities.",
                losses[0][:8],
                [_tok().decode([i]).strip()[:10] or "space" for i in first_ids[1:9]],
            ),
            _bars(
                "Mean loss in one unit",
                "All values use nats per scored GPT-2 token, weighted by token count.",
                [np.mean(uni), np.mean(bi), all_loss.mean()],
                ["unigram", "bigram", "GPT-2"],
            ),
            _bars(
                "Perplexity of those same targets",
                "Each value exponentiates the corresponding mean token loss; models see the same"
                " held-out positions.",
                ppl,
                ["unigram", "bigram", "GPT-2"],
                notes=("%d scored tokens" % len(all_loss),),
            ),
            _curve(
                "Uniform-choice analytic reference",
                "For k equally likely choices, perplexity equals k; this is a formula, not a"
                " training result.",
                np.arange(1, 17),
                ["perplexity"],
                x=np.arange(1, 17),
            ),
        ],
        source="Collected TinyShakespeare; pretrained GPT-2",
        settings={"baseline_train_tokens": len(train_ids), "scored_tokens": len(all_loss)},
        **_text_meta(
            evaluation=(
                "Eight held-out excerpts; same token positions and token-weighted aggregation"
            )
        ),
    )


def beam_search(ids, width=4, steps=5):
    beams = [(list(ids), 0.0)]
    history, kept = [], []
    for _ in range(steps):
        candidates = []
        for seq, score in beams:
            lp = np.log(np.maximum(gpt2.softmax(gpt2_hooks.forward(seq)["logits"][-1]), 1e-300))
            for token in np.argsort(-lp)[:width]:
                candidates.append((seq + [int(token)], score + float(lp[token])))
        candidates.sort(key=lambda item: item[1], reverse=True)
        history.append([c[1] for c in candidates])
        beams = candidates[:width]
        kept.append([c[1] for c in beams])
    return beams, history, kept


def _beam():
    ids = _tok().encode("The best way to learn is")
    runs = [beam_search(ids, w, 5) for w in (1, 2, 4, 8)]
    scores = [run[0][0][1] for run in runs]
    matrix = np.full((5, 8), np.nan)
    for i, row in enumerate(runs[3][2]):
        matrix[i, : len(row)] = row
    notes = tuple(_tok().decode(run[0][0][0][len(ids) :]).strip()[:50] for run in runs[:3])
    return _study(
        "Beam search",
        "GPT-2 small · five generated tokens",
        [
            _bars(
                "Candidate continuations",
                "GPT-2 assigns a probability to every extension of the prefix.",
                gpt2.softmax(gpt2_hooks.forward(ids)["logits"][-1])[
                    np.argsort(-gpt2_hooks.forward(ids)["logits"][-1])[:8]
                ],
                [
                    _tok().decode([int(t)]).strip()[:10]
                    for t in np.argsort(-gpt2_hooks.forward(ids)["logits"][-1])[:8]
                ],
            ),
            _matrix(
                "Keep eight hypotheses",
                "Each row is a decoding step; colour shows cumulative log probability.",
                np.nan_to_num(matrix, nan=np.nanmin(matrix)),
                notes=("Eight candidates survive each pruning step.",),
            ),
            _bars(
                "Width changes the search",
                "Scores are the sums of the five actual token log probabilities.",
                scores,
                ["greedy", "beam 2", "beam 4", "beam 8"],
            ),
            _curve(
                "Best score through time",
                "A wider beam retains alternatives that a greedy choice discards.",
                [[row[0] for row in run[2]] for run in runs],
                ["1", "2", "4", "8"],
                x=np.arange(1, 6),
                notes=notes,
            ),
        ],
        source="Pinned GPT-2",
        settings={"steps": 5, "widths": [1, 2, 4, 8]},
    )


def speculative_distribution(p, q):
    """Exact one-proposal speculative output mass, including rejection correction."""
    p, q = np.asarray(p), np.asarray(q)
    accepted = np.minimum(p, q)
    residual = np.maximum(p - q, 0)
    rejected = 1 - accepted.sum()
    correction = residual / max(residual.sum(), 1e-300)
    return accepted + rejected * correction


def _speculative():
    import time

    start_time = time.perf_counter()
    ids = _tok().encode("The capital of France is")
    rng = np.random.default_rng(SEED)
    accepted, expected, errors, seq = [], [], [], list(ids)
    target_calls = 0
    while len(accepted) < 12:
        draft_prefix = list(seq)
        proposals = []
        draft_probs = []
        for _ in range(3):
            q = gpt2.softmax(gpt2_hooks.forward(draft_prefix, n_layers=4)["logits"][-1])
            token = int(rng.choice(len(q), p=q))
            proposals.append(token)
            draft_probs.append(q)
            draft_prefix.append(token)
        # A single target call verifies every proposed token in parallel.
        verified = gpt2_hooks.forward(seq + proposals)["logits"]
        target_calls += 1
        all_accepted = True
        for token, q in zip(proposals, draft_probs):
            p = gpt2.softmax(verified[len(seq) - 1])
            expected.append(np.minimum(p, q).sum())
            errors.append(np.max(np.abs(speculative_distribution(p, q) - p)))
            good = rng.random() < min(1.0, p[token] / max(q[token], 1e-300))
            accepted.append(good)
            if good:
                seq.append(token)
            else:
                correction = np.maximum(p - q, 0)
                correction /= correction.sum()
                seq.append(int(rng.choice(len(correction), p=correction)))
                all_accepted = False
                break
            if len(accepted) >= 12:
                break
        if all_accepted and len(accepted) < 12:
            p = gpt2.softmax(verified[-1])
            seq.append(int(rng.choice(len(p), p=p)))
    spec_seconds = time.perf_counter() - start_time
    baseline_sequence = list(ids)
    baseline_start = time.perf_counter()
    for _ in range(len(seq) - len(ids)):
        logits = gpt2_hooks.forward(baseline_sequence)["logits"][-1]
        baseline_sequence.append(int(logits.argmax()))
    baseline_seconds = time.perf_counter() - baseline_start
    accepted = np.asarray(accepted[:12])
    expected = np.asarray(expected[:12])
    errors = errors[:12]
    p = gpt2.softmax(gpt2_hooks.forward(ids)["logits"][-1])
    q = gpt2.softmax(gpt2_hooks.forward(ids, n_layers=4)["logits"][-1])
    top = np.argsort(-p)[:4]
    return _study(
        "Speculative decoding",
        "GPT-2 block verifier · four-layer GPT-2 draft",
        [
            _bars(
                "Draft versus verifier",
                "Draft three tokens; one target pass verifies their conditional probabilities.",
                np.r_[p[top], q[top]],
                ["P1", "P2", "P3", "P4", "Q1", "Q2", "Q3", "Q4"],
            ),
            _bars(
                "Accepted or corrected",
                "Accept with min(1,p/q); rejection samples the residual and discards later drafts.",
                accepted,
                [str(i + 1) for i in range(12)],
                notes=("Observed acceptance: %.1f%%" % (accepted.mean() * 100),),
            ),
            _curve(
                "Distribution identity",
                "Accepted mass plus the rejection correction equals the target distribution.",
                [errors, np.zeros(12)],
                ["max absolute error", "zero"],
                notes=("Largest identity error: %.2g" % max(errors),),
            ),
            _bars(
                "Verifier work avoided",
                "These are actual target forward calls, not a wall-clock speed-up claim.",
                [len(seq) - len(ids), target_calls],
                ["generated tokens", "target calls"],
                notes=(
                    "Measured baseline/spec time: %.2fx" % (baseline_seconds / spec_seconds),
                    _tok().decode(seq[len(ids) :]).strip()[:24],
                ),
            ),
        ],
        source="Pinned GPT-2",
        settings={"draft_layers": 4, "block": 3, "proposals": 12},
        timing_seconds={"baseline": baseline_seconds, "speculative": spec_seconds},
        limitations=["Target-call reduction does not imply wall-clock acceleration in NumPy"],
    )


def _penalties():
    seq = _tok().encode("I think that I think that I think that")
    logits = gpt2_hooks.forward(seq)["logits"][-1].astype(float)
    counts = np.bincount(seq, minlength=len(logits))
    frequency = logits - 0.7 * counts
    repetition = np.where(counts > 0, np.where(logits > 0, logits / 1.3, logits * 1.3), logits)
    blocked = logits.copy()
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        if (a, b) == tuple(seq[-2:]):
            blocked[c] = -np.inf
    probs = [gpt2.softmax(v) for v in (logits, repetition, frequency, blocked)]
    top = np.argsort(-probs[0])[:8]
    labels = [_tok().decode([int(t)]).strip()[:10] or "space" for t in top]
    return _study(
        "Repetition and penalties",
        "GPT-2 small · repeated-prefix experiment",
        [
            _bars(
                "Before a penalty",
                "The input repeats a phrase; these are the next-token probabilities.",
                probs[0][top],
                labels,
            ),
            _bars(
                "Repetition penalty",
                "Divide positive seen-token logits by 1.3; multiply negative ones by 1.3.",
                probs[1][top],
                labels,
            ),
            _bars(
                "Frequency penalty",
                "Subtract 0.7 times the count of each token, then normalize again.",
                probs[2][top],
                labels,
            ),
            _bars(
                "Block repeated trigrams",
                "Tokens completing an already observed trigram receive zero probability.",
                probs[3][top],
                labels,
                notes=("Banned tokens: %d" % np.isneginf(blocked).sum(),),
            ),
        ],
        source="Pinned GPT-2",
        settings={"repeat_penalty": 1.3, "frequency_penalty": 0.7},
    )


def _constrained():
    prefix = "Return only two digits, a comma, and two digits: "
    seq = _tok().encode(prefix)
    masses, choices, probabilities = [], [], []
    for char_class in ("digit", "digit", "comma", "digit", "digit"):
        p = gpt2.softmax(gpt2_hooks.forward(seq)["logits"][-1])
        allowed = (
            [
                i
                for i in range(gpt2.N_VOCAB)
                if (
                    len(_tok().decode([i])) == 1
                    and _tok().decode([i]).isascii()
                    and _tok().decode([i]).isdigit()
                )
            ]
            if char_class == "digit"
            else _tok().encode(",")
        )
        mask = np.zeros_like(p, dtype=bool)
        mask[allowed] = True
        masses.append(float(p[mask].sum()))
        constrained = np.where(mask, p, 0)
        constrained /= constrained.sum()
        token = int(np.argmax(constrained))
        choices.append(token)
        probabilities.append(constrained[allowed[:10]])
        seq.append(token)
    return _study(
        "Constrained decoding",
        "GPT-2 small · grammar: DD,DD",
        [
            _matrix(
                "The grammar mask",
                "Each state permits one ASCII digit, or the separator comma.",
                [[1] * 10, [1] * 10, [0] * 10, [1] * 10, [1] * 10],
                notes=("Comma is the only token allowed at state 3.",),
            ),
            _bars(
                "Mass that survives",
                "A grammar can remove most of the model probability mass.",
                masses,
                ["digit 1", "digit 2", "comma", "digit 3", "digit 4"],
            ),
            _bars(
                "Mass removed",
                "Allowed probabilities are divided by the surviving mass.",
                1 - np.array(masses),
                ["digit 1", "digit 2", "comma", "digit 3", "digit 4"],
            ),
            _bars(
                "A valid generated string",
                "Every emitted token obeys the current finite-state grammar.",
                [1] * 5,
                [_tok().decode([i]) for i in choices],
                notes=("Output: " + _tok().decode(choices),),
            ),
        ],
        source="Pinned GPT-2",
        settings={"grammar": "DD,DD"},
    )


def _contrastive():
    ids = _tok().encode("A telescope is useful because it allows us to")
    expert = gpt2_hooks.forward(ids)["logits"][-1]
    amateur = gpt2_hooks.forward(ids, n_layers=4)["logits"][-1]
    ep, ap = gpt2.softmax(expert), gpt2.softmax(amateur)
    plausible = ep >= 0.1 * ep.max()
    score = np.log(np.maximum(ep, 1e-300)) - np.log(np.maximum(ap, 1e-300))
    contrast = gpt2.softmax(np.where(plausible, score, -np.inf))
    top = np.argsort(-ep)[:8]
    labels = [_tok().decode([int(t)]).strip()[:12] for t in top]
    return _study(
        "Contrastive decoding",
        "GPT-2 small · four-layer amateur",
        [
            _bars(
                "The expert",
                "The full twelve-layer model supplies the plausible token set.",
                ep[top],
                labels,
            ),
            _bars(
                "The amateur",
                "Stopping the same checkpoint at layer four gives a weaker distribution.",
                ap[top],
                labels,
            ),
            _bars(
                "Log probability difference",
                "Subtract the amateur log probability from the expert log probability.",
                score[top],
                labels,
            ),
            _bars(
                "After the plausibility filter",
                "Tokens below 10% of the expert maximum are removed before normalization.",
                contrast[top],
                labels,
                notes=(
                    "Allowed tokens: %d" % plausible.sum(),
                    "Selected: " + _tok().decode([int(contrast.argmax())]).strip(),
                ),
            ),
        ],
        source="Pinned GPT-2",
        settings={"amateur_layers": 4, "alpha": 0.1},
    )


def preference_loss(P, features, winners, losers):
    """Bradley–Terry cross entropy and the linear reward gradient."""
    diff = features[winners] - features[losers]
    gap = diff @ P["w"]
    loss = np.logaddexp(0, -gap).mean()
    grad = -(gpt2.softmax(np.stack([np.zeros_like(gap), -gap], -1))[:, 1, None] * diff).mean(0)
    return float(loss), {"w": grad}


@lru_cache(maxsize=1)
def _preferences():
    """Automatic observed-next-word preferences from real text, never human labels."""
    from collections import Counter

    train_words = re.findall(r"[a-z]+", _corpus()["train"].lower())
    test_words = re.findall(r"[a-z]+", _corpus()["test"].lower())
    following = [b for a, b in itertools.pairwise(train_words) if a == "the"]
    frequencies = Counter(following)
    words = ["the"] + [w for w, _ in frequencies.most_common() if w != "the"][:47]
    ix = {w: i for i, w in enumerate(words)}
    observed = np.asarray([ix[w] for w in following if w in ix])
    held = np.asarray([ix[b] for a, b in itertools.pairwise(test_words) if a == "the" and b in ix])
    if len(held) < 10:
        raise ValueError("Insufficient held-out real next-word examples")
    marginal = Counter(train_words)
    reference = np.asarray([marginal[w] + 1 for w in words], float)
    reference /= reference.sum()
    held_prob = (np.bincount(held, minlength=len(words)) + 0.5) / (len(held) + 0.5 * len(words))
    # Features describe actual candidate words. The reward deliberately has limited capacity.
    char_count = Counter("".join(train_words))
    raw = np.asarray(
        [
            [
                len(w),
                sum(c in "aeiou" for c in w) / len(w),
                np.mean([np.log1p(char_count[c]) for c in w]),
            ]
            for w in words
        ]
    )
    features = np.c_[(raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-8), np.ones(len(words))]
    rng = np.random.default_rng(SEED)
    winners = rng.choice(observed, 800, replace=True)
    losers = rng.integers(len(words), size=len(winners))
    keep = winners != losers
    winners, losers = winners[keep], losers[keep]
    P = {"w": rng.normal(0, 0.1, features.shape[1])}
    opt = Adam(P, 0.03)
    loss = []
    for _ in range(250):
        value, g = preference_loss(P, features, winners, losers)
        opt.step(P, g)
        loss.append(value)
    held_losers = rng.integers(len(words), size=len(held))
    valid = held != held_losers
    reward = features @ P["w"]
    evidence = _save_weights(
        "automatic-reward",
        P,
        history={
            "loss": loss,
            "train_winner": winners,
            "train_loser": losers,
            "heldout_observed": held,
        },
        steps=250,
        prompt="the",
        candidates=words,
        preference_rule="observed corpus continuation preferred to sampled alternative",
    )
    return dict(
        features=features,
        x=np.arange(len(words)),
        true=np.log(held_prob),
        reward=reward,
        ref=reference,
        loss=np.asarray(loss),
        winners=winners,
        losers=losers,
        P=P,
        words=words,
        test_winners=held[valid],
        test_losers=held_losers[valid],
        artifact=evidence,
        observed_train=len(observed),
        observed_test=len(held),
    )


def policy_grads(logits, reference, reward, beta):
    p = gpt2.softmax(logits)
    log_ratio = np.log(np.maximum(p / reference, 1e-300))
    objective = p @ reward - beta * (p @ log_ratio)
    score = reward - beta * (log_ratio + 1)
    gradient = p * (score - p @ score)
    return float(objective), gradient


def _policy_trace(beta=0.15, steps=180, ppo=True):
    """Exact categorical PPO surrogate over a finite completion bandit."""
    d = _preferences()
    z = np.log(d["ref"])
    rows, distributions = [], []
    for _ in range(steps):
        old = gpt2.softmax(z)
        adv = d["reward"] - old @ d["reward"]
        for _epoch in range(4 if ppo else 1):
            p = gpt2.softmax(z)
            ratio = p / old
            active = ~(((adv > 0) & (ratio > 1.2)) | ((adv < 0) & (ratio < 0.8)))
            raw = old * ratio * adv * active
            grad = raw - p * raw.sum()
            _, klg = policy_grads(z, d["ref"], np.zeros_like(z), beta)
            z += 0.8 * (grad + klg)
        p = gpt2.softmax(z)
        rows.append([p @ d["reward"], p @ d["true"], p @ np.log(np.maximum(p / d["ref"], 1e-300))])
        distributions.append(p)
    return np.asarray(rows).T, np.asarray(distributions)


def _reward_model():
    d = _preferences()
    gap = np.linspace(-6, 6, 101)
    accuracy = (d["reward"][d["test_winners"]] > d["reward"][d["test_losers"]]).mean()
    return _study(
        "Learning an automatic preference score",
        "Real text: observed word after 'the' versus sampled alternative",
        [
            _curve(
                "Pairwise logistic link",
                "This analytic curve maps score differences to a preference probability.",
                1 / (1 + np.exp(-gap)),
                ["P(preferred)"],
                x=gap,
            ),
            _curve(
                "Actual reward-model training",
                "A linear model fits automatic observed-versus-alternative word pairs.",
                d["loss"],
                ["training pair loss"],
            ),
            _curve(
                "Held-out frequency and learned score",
                "Candidate IDs denote real words; the corpus frequency is an evaluation score, not"
                " human utility.",
                [d["true"], d["reward"]],
                ["held-out log frequency", "learned linear score"],
                x=d["x"],
            ),
            _bars(
                "Held-out pair ranking",
                "Pairs use an untouched corpus partition; noisy automatic labels need not be"
                " perfectly ranked.",
                [accuracy, 1 - accuracy],
                ["correct order", "other order"],
            ),
        ],
        source="Collected TinyShakespeare automatic preferences",
        settings={"steps": 250, "candidate_words": d["words"]},
        **_text_meta(
            weights_artifacts=[d["artifact"]],
            preference_rule="observed next word versus random candidate; not human feedback",
        ),
    )


def _ppo():
    d = _preferences()
    runs = [_policy_trace(b) for b in (0.0, 0.15, 1.0)]
    labels = ["beta0", "beta.15", "beta1"]
    artifacts = [
        _save_weights(
            "categorical-ppo-%d" % i,
            {"logits": np.log(np.maximum(r[1][-1], 1e-300))},
            history={"metrics": r[0], "probabilities": r[1]},
            steps=180,
            beta=b,
        )
        for i, (r, b) in enumerate(zip(runs, (0.0, 0.15, 1.0)))
    ]
    return _study(
        "PPO with automatic text preferences",
        "TinyShakespeare word-choice bandit · not human-feedback RLHF",
        [
            _curve(
                "Optimize the learned score",
                "Categorical logits receive real clipped PPO gradient updates.",
                [r[0][0] for r in runs],
                labels,
            ),
            _curve(
                "Divergence from corpus reference",
                "A KL penalty constrains drift from training-corpus word frequencies.",
                [r[0][2] for r in runs],
                labels,
            ),
            _curve(
                "Held-out corpus score",
                "Expected log frequency after 'the' is measured on held-out text; it is not human"
                " reward.",
                [r[0][1] for r in runs],
                labels,
            ),
            _curve(
                "Final next-word distributions",
                "Candidate IDs refer to the recorded real-word vocabulary.",
                [r[1][-1] for r in runs] + [d["ref"]],
                labels + ["reference"],
                x=d["x"],
            ),
        ],
        source="Collected TinyShakespeare automatic preferences",
        settings={"steps": 180, "epochs": 4, "clip": 0.2, "candidate_words": d["words"]},
        **_text_meta(
            weights_artifacts=[d["artifact"]] + artifacts,
            preference_rule="automatic corpus continuation; no human labels",
        ),
    )


def dpo_loss(logits, reference, winners, losers, beta=0.3):
    margin = beta * (
        logits[winners] - logits[losers] - np.log(reference[winners]) + np.log(reference[losers])
    )
    factor = -beta / (1 + np.exp(np.clip(margin, -60, 60))) / len(winners)
    grad = np.zeros_like(logits)
    np.add.at(grad, winners, factor)
    np.add.at(grad, losers, -factor)
    return float(np.logaddexp(0, -margin).mean()), grad


def neural_dpo_loss(P, prompt, config, reference, winners, losers, offset):
    from . import tinylm as lm

    params = {k: lm.Var(v) for k, v in P.items()}
    logits, _ = lm._forward(params, prompt, config)
    z = logits[0, -1] + offset
    gap = 0.3 * (z[winners] - z[losers] - np.log(reference[winners]) + np.log(reference[losers]))
    loss = (1 + (-gap).exp()).log().mean()
    loss.backward()
    return float(loss.v), {k: np.zeros_like(v.v) if v.g is None else v.g for k, v in params.items()}


def _dpo():
    from . import tinylm as lm

    d = _preferences()
    cfg = lm.Config(vocab=len(d["ref"]), width=16, heads=4, kv_heads=4, ff=32)
    prompt = np.array([[0]])  # candidate vocabulary ID0 is the actual word 'the'.
    P = lm.init(cfg, SEED)
    offset = np.log(d["ref"]) - lm.forward(P, prompt, cfg)[0, -1]
    opt = Adam(P, 0.002)
    loss = []
    trace = []
    for _ in range(180):
        lo, g = neural_dpo_loss(P, prompt, cfg, d["ref"], d["winners"], d["losers"], offset)
        opt.step(P, g)
        p = gpt2.softmax(lm.forward(P, prompt, cfg)[0, -1] + offset)
        loss.append(lo)
        trace.append([p @ d["true"], p @ d["reward"], p @ np.log(np.maximum(p / d["ref"], 1e-300))])
    tr = np.asarray(trace).T
    ppo, _ = _policy_trace(0.3)
    artifact = _save_weights(
        "dpo",
        dict(P, output_offset=offset),
        history={"loss": loss, "metrics": tr},
        steps=180,
        beta=0.3,
        candidates=d["words"],
    )
    return _study(
        "DPO on real next-word preference pairs",
        "TinyShakespeare · automatic labels, not human preference judgments",
        [
            _curve(
                "Measured DPO loss",
                "A tiny transformer trains on observed-versus-alternative word pairs.",
                loss,
                ["DPO logistic loss"],
            ),
            _curve(
                "Implicit score across real candidate words",
                "Beta times log policy/reference; its scale is not a calibrated human reward.",
                0.3 * np.log(np.maximum(p / d["ref"], 1e-300)),
                ["implicit preference score"],
                x=d["x"],
            ),
            _curve(
                "Held-out text frequency objective",
                "Both use identical real-text preferences; DPO is neural, the PPO comparator"
                " categorical.",
                [tr[0], ppo[1]],
                ["DPO corpus score", "PPO corpus score"],
            ),
            _curve(
                "KL drift during actual training",
                "Only policy divergence is plotted here; reward is a different quantity.",
                tr[2],
                ["KL nats"],
            ),
        ],
        source="Collected TinyShakespeare automatic preferences",
        settings={"steps": 180, "beta": 0.3, "candidate_words": d["words"]},
        **_text_meta(weights_artifacts=[d["artifact"], artifact]),
    )


def _reward_hacking():
    d = _preferences()
    logits = np.log(d["ref"])
    distributions = []
    trace = []
    # Execute the exponentiated-gradient steps; preserve every computed iterate.
    for step in range(241):
        p = gpt2.softmax(logits)
        distributions.append(p)
        trace.append([p @ d["reward"], p @ d["true"]])
        if step < 240:
            logits += (8 / 240) * d["reward"]
    dist = np.asarray(distributions)
    values = np.asarray(trace).T
    pressure = np.linspace(0, 8, 241)
    peak = int(values[1].argmax())
    gap = float(values[1, peak] - values[1, -1])
    outcome = (
        "Held-out score falls after its peak."
        if gap > 0.01
        else "No material post-peak decline in this run."
    )
    artifact = _save_weights(
        "reward-pressure",
        {"logits": logits},
        history={"probabilities": dist, "metrics": values},
        steps=240,
        step_size=8 / 240,
    )
    return _study(
        "Does optimizing a proxy harm the real-text objective?",
        "TinyShakespeare · measured policy updates; no failure outcome assumed",
        [
            _curve(
                "Proxy and held-out corpus score",
                "A limited linear scorer fits automatic preferences from the training split.",
                [d["reward"], d["true"]],
                ["learned proxy", "held-out log frequency"],
                x=d["x"],
            ),
            _curve(
                "Actual exponentiated-gradient updates",
                "Every iterate is computed; held-out text provides a separate evaluation"
                " objective.",
                values,
                ["proxy expectation", "held-out expectation"],
                x=pressure,
                notes=(outcome,),
            ),
            _matrix(
                "Recorded next-word probability mass",
                "Rows are saved updates; columns are actual candidate word IDs.",
                dist[::6],
            ),
            _bars(
                "Held-out outcome, including failed hypotheses",
                "Optimization may or may not overfit this proxy; these signed values are measured.",
                [values[1, 0], values[1, peak], values[1, -1]],
                ["initial", "best observed", "final"],
                notes=(outcome,),
            ),
        ],
        source="Collected TinyShakespeare automatic preferences",
        settings={"steps": 240, "pressure": 8.0, "candidate_words": d["words"]},
        **_text_meta(weights_artifacts=[d["artifact"], artifact]),
    )


def superposition_loss(P, x):
    h = x @ P["W"].T
    pre = h @ P["W"] + P["b"]
    out = np.maximum(pre, 0)
    delta = 2 * (out - x) * (pre > 0) / len(x)
    grad = (x.T @ (delta @ P["W"].T)).T + h.T @ delta
    return float(((out - x) ** 2).sum(1).mean()), {"W": grad, "b": delta.sum(0)}


def _superposition():
    x, _, train_meta = realdata.mnist_split(2048, "train", 14, SEED)
    xt, _, test_meta = realdata.mnist_split(512, "test", 14, SEED + 1)
    locations = ((2, 2), (2, 9), (6, 6), (9, 2), (9, 9))

    def features(images):
        images = images.reshape(-1, 14, 14)
        return np.stack([images[:, i : i + 3, j : j + 3].mean((1, 2)) for i, j in locations], 1)

    raw, held = features(x), features(xt)
    vectors = []
    losses = []
    held_loss = []
    artifacts = []
    actual = []
    for q in (0.1, 0.5, 0.9, 0.98):
        threshold = np.quantile(raw, q, axis=0)
        data = np.where(raw > threshold, raw, 0)
        test = np.where(held > threshold, held, 0)
        rng = np.random.default_rng(SEED)
        P = {"W": rng.normal(0, 0.3, (2, 5)), "b": np.zeros(5)}
        opt = Adam(P, 0.01)
        trace = []
        for _ in range(800):
            ix = rng.integers(len(data), size=128)
            lo, g = superposition_loss(P, data[ix])
            opt.step(P, g)
            trace.append(lo)
        vectors.append(P["W"].copy())
        losses.append(trace)
        held_loss.append(superposition_loss(P, test)[0])
        actual.append(float((data == 0).mean()))
        artifacts.append(
            _save_weights(
                "mnist-superposition-%s" % q,
                P,
                history={"training_loss": trace, "thresholds": threshold},
                dataset=train_meta,
                steps=800,
            )
        )
    return _study(
        "Superposition of MNIST-derived patch features",
        "Five measured image patches · two learned hidden coordinates",
        [
            _panel(
                "Low-threshold feature directions",
                "Each direction is learned from actual MNIST patch intensities, not random"
                " synthetic feature vectors.",
                "points",
                xy=vectors[0].T,
                classes=np.arange(5),
            ),
            _panel(
                "High-threshold feature directions",
                "Thresholds fitted on training images vary sparsity; test images use those same"
                " thresholds.",
                "points",
                xy=vectors[-1].T,
                classes=np.arange(5),
            ),
            _matrix(
                "Measured feature interference",
                "The Gram matrix comes from the final learned two-dimensional feature directions.",
                vectors[-1].T @ vectors[-1],
            ),
            _curve(
                "Actual reconstruction training",
                "Mostly-zero inputs lower raw error; held-out losses and sparsities are recorded in"
                " the manifest.",
                losses,
                ["quantile .10", "quantile .50", "quantile .90", "quantile .98"],
            ),
        ],
        source="MNIST measured patch intensities",
        settings={"steps": 800, "patch_locations": locations, "quantiles": [0.1, 0.5, 0.9, 0.98]},
        dataset="MNIST",
        dataset_provenance={"train": train_meta, "test": test_meta},
        weights_artifacts=artifacts,
        heldout_loss=held_loss,
        actual_zero_fraction=actual,
    )


@lru_cache(maxsize=1)
def _activation_samples():
    text = _corpus()["train"]
    starts = np.linspace(0, len(text) - 128, 24).astype(int)
    texts = [text[i : i + 128] for i in starts]
    return texts, [_forward(t, record=True) for t in texts]


def sparse_loss(P, x, penalty=0.01):
    pre = x @ P["E"] + P["b"]
    h = np.maximum(pre, 0)
    out = h @ P["D"] + P["c"]
    delta = 2 * (out - x) / (len(x) * x.shape[1])
    dh = (delta @ P["D"].T + penalty / (len(x) * h.shape[1])) * (pre > 0)
    return float(np.mean((out - x) ** 2) + penalty * h.mean()), dict(
        E=x.T @ dh, b=dh.sum(0), D=h.T @ delta, c=delta.sum(0)
    )


def _sae():
    texts, records = _activation_samples()
    raw = np.concatenate([r["resid"][6] for r in records]).astype(float)
    testtext = _corpus()["test"]
    heldtexts = [testtext[i : i + 128] for i in np.linspace(0, len(testtext) - 128, 8).astype(int)]
    heldraw = np.concatenate([_forward(t, record=True)["resid"][6] for t in heldtexts]).astype(
        float
    )
    mean, scale = raw.mean(0), max(raw.std(), 1e-6)
    x = (raw - mean) / scale
    xt = (heldraw - mean) / scale
    rng = np.random.default_rng(SEED)
    P = dict(
        E=rng.normal(0, 0.015, (768, 896)),
        b=np.full(896, -0.1),
        D=rng.normal(0, 0.015, (896, 768)),
        c=np.zeros(768),
    )
    opt = Adam(P, 0.003)
    losses = []
    fractions = []
    for _ in range(180):
        ix = rng.integers(len(x), size=32)
        lo, g = sparse_loss(P, x[ix], 2.0)
        opt.step(P, g)
        losses.append(lo)
        fractions.append(np.mean(x[ix] @ P["E"] + P["b"] > 0))
    act = np.maximum(x @ P["E"] + P["b"], 0)
    ha = np.maximum(xt @ P["E"] + P["b"], 0)
    recon = ha @ P["D"] + P["c"]
    labels = [_tok().decode([i]) for t in texts for i in _tok().encode(t)]
    selected = np.argsort(-act.max(0))[:8]
    top = [labels[int(act[:, j].argmax())].strip()[:10] or "space" for j in selected]
    artifact = _save_weights(
        "real-text-sae",
        dict(P, center=mean, scale=np.array(scale)),
        history={"loss": losses, "active_fraction": fractions},
        steps=180,
    )
    return _study(
        "Sparse features of real-text activations",
        "GPT-2 layer six · trained on collected Shakespeare passages",
        [
            _curve(
                "Train an overcomplete dictionary",
                "The autoencoder optimizes actual GPT-2 residuals from the training text split.",
                losses,
                ["reconstruction + sparsity loss"],
            ),
            _curve(
                "Training activation fraction",
                "ReLU activation sparsity is measured at each optimizer step.",
                fractions,
                ["active fraction"],
            ),
            _bars(
                "Observed top-activating tokens",
                "One top token does not establish a unique semantic concept.",
                act[:, selected].max(0),
                top,
            ),
            _bars(
                "Untouched text reconstruction MSE",
                "Centering and scale are fitted only on training activations.",
                [np.mean((recon - xt) ** 2)],
                ["held-out MSE"],
                notes=(
                    "Held-out active fraction: %.3f" % np.mean(ha > 0),
                    "Training-dead fraction: %.3f" % np.mean(act.max(0) == 0),
                ),
            ),
        ],
        source="GPT-2 activations of collected TinyShakespeare",
        settings={"layer": 6, "features": 896, "steps": 180},
        **_text_meta(
            weights_artifacts=[artifact],
            train_residuals=len(x),
            heldout_residuals=len(xt),
            limitations=["Small corpus subset; no monosemanticity claim"],
        ),
    )


def _probes():
    def data(split, count):
        text = _corpus()[split]
        features = [[] for _ in range(13)]
        labels = []
        controls = []
        for start in np.linspace(0, len(text) - 160, count).astype(int):
            ids = _tok().encode(text[start : start + 160])
            r = gpt2_hooks.forward(ids, record=True)
            for i in range(1, len(ids)):
                previous = _tok().decode([ids[i - 1]])
                labels.append(float(any(ch in previous for ch in ".,;:!?")))
                controls.append(float((ids[i] * 2654435761 % 11) < 5))
                for l in range(13):
                    features[l].append(r["resid"][l, i])
        return [np.asarray(v) for v in features], np.asarray(labels), np.asarray(controls)

    train, y, cy = data("train", 24)
    test, yt, ct = data("test", 8)
    accuracy = []
    control = []
    weights = {}

    def balanced(pred, target):
        return np.mean([(pred[target == v] == v).mean() for v in (0, 1)])

    for layer, (tr, te) in enumerate(zip(train, test)):
        mean, scale = tr.mean(0), tr.std(0) + 0.1
        tr = np.c_[(tr - mean) / scale, np.ones(len(tr))]
        te = np.c_[(te - mean) / scale, np.ones(len(te))]
        for key, target, testtarget, out in (
            ("punctuation", y, yt, accuracy),
            ("token_control", cy, ct, control),
        ):
            w = np.zeros(tr.shape[1])
            sample = np.where(
                target > 0, 0.5 / max(target.mean(), 1e-6), 0.5 / max(1 - target.mean(), 1e-6)
            )
            for _ in range(120):
                p = 1 / (1 + np.exp(np.clip(-tr @ w, -60, 60)))
                w -= 0.1 * (tr.T @ ((p - target) * sample) / len(tr) + 0.05 * w)
            out.append(balanced(te @ w > 0, testtarget))
            weights[key + str(layer)] = w
    artifact = _save_weights(
        "context-probes",
        weights,
        history={"balanced_accuracy": accuracy, "token_control": control},
        steps=120,
    )
    return _study(
        "Probing context in collected text",
        "GPT-2 · does the preceding token contain punctuation?",
        [
            _bars(
                "Context labels come from the source",
                "No generated sentence template: labels are derived from actual neighboring"
                " tokens.",
                [y.mean(), yt.mean()],
                ["train positive fraction", "test positive fraction"],
            ),
            _curve(
                "Held-out balanced accuracy by layer",
                "Every layer gets its own linear readout; label balancing prevents a majority-class"
                " score.",
                accuracy,
                ["preceding punctuation"],
                x=np.arange(13),
            ),
            _curve(
                "Token-identity control",
                "A deterministic current-token label measures easily decoded lexical information.",
                control,
                ["token control"],
                x=np.arange(13),
            ),
            _curve(
                "Decodability is not causal use",
                "Layer0 is an input-embedding baseline; contextual association is not proof of a"
                " circuit.",
                [accuracy, control],
                ["preceding punctuation", "current-token control"],
                x=np.arange(13),
            ),
        ],
        source="Collected TinyShakespeare held-out context labels",
        settings={"train_excerpts": 24, "test_excerpts": 8, "steps": 120},
        **_text_meta(
            weights_artifacts=[artifact], target="previous GPT2 token contains one of . , ; : ! ?"
        ),
    )


def _patching_data():
    observed = _tok().encode(_corpus()["test"][:160])
    clean = observed[:9]
    target = observed[9]
    replacement = _tok().encode(_corpus()["train"][200:300])
    corrupt = clean.copy()
    corrupt[2:4] = replacement[:2]
    cr = gpt2_hooks.forward(clean, record=True)
    baseline = gpt2.softmax(cr["logits"][-1])[target]
    dirty = gpt2.softmax(gpt2_hooks.forward(corrupt)["logits"][-1])[target]
    matrix = np.zeros((13, len(clean)))
    for layer in range(13):
        for token in range(len(clean)):
            result = gpt2_hooks.forward(
                corrupt, patches={("resid", layer, token): cr["resid"][layer, token]}
            )
            matrix[layer, token] = gpt2.softmax(result["logits"][-1])[target]
    return clean, corrupt, cr, target, float(baseline), float(dirty), matrix


@lru_cache(maxsize=1)
def _patching_cached():
    return _patching_data()


def _patching():
    clean, corrupt, _, target, baseline, dirty, matrix = _patching_cached()
    gap = baseline - dirty
    recovery = (matrix - dirty) / (gap if abs(gap) > 1e-12 else 1e-12)
    return _study(
        "Activation patching of a real passage",
        "GPT-2 · recorded next token in held-out Shakespeare",
        [
            _bars(
                "An actual recorded continuation",
                "Corruption replaces two context tokens; target is the next token in the collected"
                " passage.",
                [baseline, dirty],
                ["original excerpt", "corrupted context"],
                notes=("Target: " + repr(_tok().decode([target])),),
            ),
            _matrix(
                "Replace one residual vector",
                "Each cell copies an original-context vector into the corrupted forward pass.",
                matrix,
            ),
            _matrix(
                "Recovery relative to this probability gap",
                "Negative or excessive recovery remains visible; the context was not selected to"
                " guarantee a large gap.",
                recovery,
            ),
            _bars(
                "Strongest patch at each depth",
                "This local intervention is not evidence of a general fact-storage location.",
                recovery.max(1),
                [str(i) for i in range(13)],
            ),
        ],
        source="Pretrained GPT-2 on collected held-out text",
        settings={
            "clean": _tok().decode(clean),
            "corrupted": _tok().decode(corrupt),
            "target": _tok().decode([target]),
        },
        **_text_meta(evaluation="fixed first nine GPT2 tokens of test split; observed tenth token"),
    )


def rank_one_edit(key, delta, background, ridge=0.1):
    """Minimum covariance-weighted rank-one map constrained to key @ edit = delta.

    Woodbury keeps the solve in sample space, avoiding a 3072-square inverse.
    """
    key = np.asarray(key, float)
    background = np.asarray(background, float)
    h = background / np.sqrt(len(background))
    invkey = (key - h.T @ np.linalg.solve(ridge * np.eye(len(h)) + h @ h.T, h @ key)) / ridge
    return np.outer(invkey, delta) / float(key @ invkey)


def _knowledge_edit():
    clean, _, cr, observed, baseline, _, patch = _patching_cached()
    layer = int(np.argmax(patch[1:12].max(1)))
    candidates = _tok().encode(_corpus()["train"][400:500])
    target = next(i for i in candidates if i != observed)
    original = gpt2.softmax(cr["logits"][-1])
    key = cr["mlp_hidden"][layer, -1].astype(float)
    _, records = _activation_samples()
    background = np.concatenate([r["mlp_hidden"][layer] for r in records])
    direction = (
        gpt2.weights()["wte.weight"][target] - gpt2.weights()["wte.weight"][observed]
    ).astype(float)
    direction /= max(np.linalg.norm(direction), 1e-12)
    name = "h.%d.mlp.c_proj.weight" % layer
    base = gpt2.weights()[name]
    scales = np.array([0, 2, 5, 10, 20, 40.0])
    rows = []
    side = []
    hold = _tok().encode(_corpus()["test"][300:430])
    control, ctarget = hold[:9], hold[9]
    for scale in scales:
        edit = rank_one_edit(key, direction * scale, background)
        changed = gpt2.softmax(
            gpt2_hooks.forward(clean, weight_edits={name: base + edit})["logits"][-1]
        )
        rows.append([changed[target], changed[observed]])
        side.append(
            gpt2.softmax(
                gpt2_hooks.forward(control, weight_edits={name: base + edit})["logits"][-1]
            )[ctarget]
        )
    return _study(
        "Rank-one continuation editing",
        "GPT-2 · real passage, analytic weight intervention",
        [
            _matrix(
                "Measure an influential depth",
                "A fixed real-text continuation provides the activation-patching probe.",
                patch,
            ),
            _bars(
                "Before the intervention",
                "The alternative token is a counterfactual target, not a newly verified fact.",
                [baseline, original[target]],
                ["observed source token", "alternative token"],
            ),
            _curve(
                "Measured response to edit strength",
                "A covariance-weighted rank-one change is evaluated, not an assumed knowledge"
                " edit.",
                np.asarray(rows).T,
                ["alternative token", "observed source token"],
                x=scales,
            ),
            _curve(
                "An untouched source passage",
                "Check a separate held-out continuation for collateral change.",
                side,
                ["observed control token"],
                x=scales,
            ),
        ],
        source="Collected TinyShakespeare and pretrained GPT-2",
        settings={"layer": layer, "scales": scales.tolist(), "target": _tok().decode([target])},
        **_text_meta(
            limitations=[
                "Analytic ROME-style local update; not a factual knowledge-training benchmark."
            ]
        ),
    )


def _ablation():
    source = _corpus()["test"]
    prompts = []
    changes = np.zeros((12, 12))
    baselines = []
    for start in (0, 300):
        observed = _tok().encode(source[start : start + 160])
        ids, target = observed[:12], observed[12]
        prompts.append(dict(context=_tok().decode(ids), target=_tok().decode([target])))
        baseline = -np.log(max(gpt2.softmax(gpt2_hooks.forward(ids)["logits"][-1])[target], 1e-300))
        baselines.append(baseline)
        for layer in range(12):
            for head in range(12):
                p = gpt2.softmax(gpt2_hooks.forward(ids, ablate={(layer, head): 0})["logits"][-1])[
                    target
                ]
                changes[layer, head] += (-np.log(max(p, 1e-300)) - baseline) / 2
    return _study(
        "Head ablation on recorded continuations",
        "GPT-2 · two fixed held-out Shakespeare excerpts",
        [
            _bars(
                "Original continuation loss",
                "Targets are the actual next tokens in the source corpus.",
                baselines,
                ["excerpt A", "excerpt B"],
            ),
            _matrix(
                "Zero one head at a time",
                "Each cell is a measured target-loss change from one of144 interventions.",
                changes,
            ),
            _bars(
                "Largest observed harmful effects",
                "Head IDs are layer*12+head; effects are local to this small evaluation set.",
                np.sort(changes.ravel())[-8:],
                [str(i) for i in np.argsort(changes.ravel())[-8:]],
            ),
            _bars(
                "Signed average effect by layer",
                "A negative change means this ablation improved the recorded-token loss.",
                changes.mean(1),
                [str(i) for i in range(12)],
            ),
        ],
        source="Collected TinyShakespeare and pretrained GPT-2",
        settings={"prompts": prompts, "ablation": "zero"},
        **_text_meta(
            limitations=["Only two fixed excerpts; no universal head-importance ranking."]
        ),
    )


def _teacher():
    from . import tinylm as lm

    seq = _text_windows(256, 16)
    c = lm.Config(vocab=_corpus()["vocab"], width=24, heads=4, kv_heads=4)
    run = lm.train(c, seq[:, :-1], seq[:, 1:], steps=350, seed=SEED, name="real-teacher-v8")
    test = _text_windows(96, 16, "test", SEED + 1)
    teacher = nll(lm.forward(run["P"], test[:, :-1], c), test[:, 1:]).mean(0)
    free = test[:, :1].copy()
    for _ in range(15):
        free = np.c_[free, lm.forward(run["P"], free, c)[:, -1].argmax(-1)]
    generated = nll(lm.forward(run["P"], free, c), test[:, 1:]).mean(0)
    artifact = _save_weights("teacher", run["P"], steps=350)
    return _study(
        "Teacher forcing on real text",
        "TinyShakespeare · NumPy character transformer",
        [
            _matrix(
                "One excerpt, shifted targets",
                "Every source character predicts the next recorded character.",
                np.array([seq[0, :-1], seq[0, 1:]]),
            ),
            _matrix(
                "Hide future characters",
                "The causal mask keeps each next-character target out of its own context.",
                lm.attention_mask(16, c).astype(float),
            ),
            _curve(
                "Measured training loss",
                "These losses come from 350 actual NumPy optimizer updates.",
                run["loss"],
                ["training nats / character"],
            ),
            _curve(
                "Recorded versus generated context",
                "Both score the recorded continuation; other generated continuations can also be"
                " valid.",
                [teacher, generated],
                ["recorded context", "greedy generated context"],
                notes=("Held-out text only; not a validity score.",),
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 350},
        **_text_meta(
            weights_artifacts=[artifact], evaluation="96 held-out excerpts; fixed recorded targets"
        ),
    )


def _packing():
    from . import tinylm as lm

    lines = [line for line in _corpus()["train"].splitlines() if 3 <= len(line) <= 80][:12]
    lengths = np.asarray([len(line) for line in lines])
    used, padded = int(lengths.sum()), int(len(lengths) * lengths.max())
    seq = _text_windows(192, 8)
    test = _text_windows(64, 8, "test", SEED + 1)
    c = lm.Config(vocab=_corpus()["vocab"], width=24)
    run = lm.train(c, seq[:, :-1], seq[:, 1:], steps=220, seed=SEED, name="real-packing-v8")
    x, y = test[:, :-1], test[:, 1:]
    px, py = np.c_[x[:32], x[32:]], np.c_[y[:32], y[32:]]
    ids = np.repeat([0, 1], 8)
    mask = np.tril(np.ones((16, 16), bool)) & (ids[:, None] == ids[None, :])
    ordinary = lm.loss(run["P"], px, py, c)
    reset_params = {k: v.copy() for k, v in run["P"].items()}
    reset_params["pos"][8:16] = reset_params["pos"][:8]
    reset = lm.loss(reset_params, px, py, c, mask=mask)
    individual = lm.loss(run["P"], x, y, c)
    return _study(
        "Packing real-text excerpts",
        "TinyShakespeare · held-out document boundaries",
        [
            _bars(
                "Lengths of twelve real text lines",
                "These source lines illustrate padding; the model test below packs fixed"
                " eight-character excerpts.",
                lengths,
                [str(i + 1) for i in range(12)],
            ),
            _bars(
                "Useful characters and padding",
                "Lengths are counted from collected text, not sampled from a length distribution.",
                [used, padded - used],
                ["source characters", "padding cells"],
            ),
            _matrix(
                "Keep excerpts independent",
                "Block-causal attention prevents one excerpt from reading another.",
                mask.astype(float),
            ),
            _bars(
                "Same held-out predictions",
                "Reset both attention and positional indices to recover separately evaluated"
                " excerpts.",
                [individual, ordinary, reset],
                ["individual", "no boundary reset", "mask + positions"],
                notes=("Reset minus individual: %.2g" % (reset - individual),),
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 220},
        **_text_meta(
            weights_artifacts=[_save_weights("packing", run["P"], steps=220)],
            evaluation="64 held-out excerpts",
        ),
    )


def _warmup():
    from . import tinylm as lm

    seq = _text_windows(192, 16)
    test = _text_windows(64, 16, "test", SEED + 1)
    c = lm.Config(vocab=_corpus()["vocab"], width=24, layers=4, placement="post")
    runs = [
        lm.train(
            c,
            seq[:, :-1],
            seq[:, 1:],
            steps=240,
            lr=0.02,
            seed=SEED,
            warmup=w,
            clip=clip,
            name="real-warmup-%s-%s-v8" % (w, clip),
        )
        for w, clip in ((0, None), (60, None), (60, 1.0))
    ]
    labels = ["fixed rate", "warmup", "warmup + clip"]
    held = [lm.loss(r["P"], test[:, :-1], test[:, 1:], c) for r in runs]
    return _study(
        "Warm-up and clipping on real text",
        "TinyShakespeare · four-layer post-LN transformer",
        [
            _curve(
                "Learning-rate policies",
                "The chosen schedules control real optimizer updates; they are not measured"
                " losses.",
                [np.full(240, 0.02), 0.02 * np.minimum(1, np.arange(1, 241) / 60)],
                ["fixed", "warmup"],
            ),
            _curve(
                "Actual minibatch loss",
                "A spike or a winner is not imposed; each curve is recorded during training.",
                [r["loss"] for r in runs],
                labels,
            ),
            _curve(
                "Measured gradient norms",
                "Norms are captured before clipping.",
                [r["grad_norm"] for r in runs],
                labels,
            ),
            _bars(
                "Held-out next-character loss",
                "All models are evaluated on the same untouched source excerpts.",
                held,
                labels,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 240, "lr": 0.02},
        **_text_meta(
            weights_artifacts=[
                _save_weights("warmup-%d" % i, r["P"], steps=240) for i, r in enumerate(runs)
            ]
        ),
    )


def _grokking():
    from . import mnist
    from .optim import AdamW

    x, y, train_meta = realdata.mnist_split(600, "train", 7, SEED)
    xt, yt, test_meta = realdata.mnist_split(1000, "test", 7, SEED + 1)
    rng = np.random.default_rng(SEED)
    layers = mnist._init(rng, (49, 64, 10))
    P = {
        "%s%d" % (key, i): v for i, layer in enumerate(layers) for key, v in zip(("W", "b"), layer)
    }
    opt = AdamW(P, 0.002, weight_decay=0.02)
    history = []
    for step in range(3001):
        if step % 50 == 0:
            params = mnist._as_layers(P, 2)
            pt = mnist._forward(params, x)[2]
            pe = mnist._forward(params, xt)[2]
            history.append(
                [
                    step,
                    (pt.argmax(1) == y).mean(),
                    (pe.argmax(1) == yt).mean(),
                    np.sqrt(sum((v * v).sum() for v in P.values())),
                ]
            )
        if step == 3000:
            break
        ix = rng.choice(len(x), 64, False)
        opt.step(P, mnist._mlp_grads(P, x[ix], np.eye(10)[y[ix]]))
    h = np.asarray(history).T
    artifact = _save_weights(
        "mnist-long-run",
        P,
        history={"training_history": h},
        dataset=train_meta,
        steps=3000,
        mnist_sha256=mnist.SHA256,
    )
    early = np.flatnonzero((h[1] > 0.95) & (h[2] < 0.2))
    claim = (
        "No delayed-generalization claim is assumed."
        if not len(early)
        else "Inspect train/test separation; one run does not establish grokking."
    )
    return _study(
        "Does longer training produce delayed generalization?",
        "MNIST · 600 train images · real NumPy MLP updates",
        [
            _panel(
                "Real handwritten inputs",
                "7x7 area-averaged MNIST images; the official test images never update weights.",
                "images",
                values=x[:6].reshape(-1, 7, 7),
                labels=[str(v) for v in y[:6]],
            ),
            _curve(
                "Training and held-out accuracy",
                "Both traces are measured every fifty actual updates; a sudden jump is not"
                " guaranteed.",
                h[1:3],
                ["600 training images", "1000 test images"],
                x=h[0],
                notes=(claim,),
            ),
            _curve(
                "Parameter norm during training",
                "AdamW decay and fitted weights are measured, not drawn from an expected"
                " transition.",
                h[3],
                ["weight norm"],
                x=h[0],
            ),
            _bars(
                "Final generalization gap",
                "This MNIST experiment tests delayed generalization; it does not reproduce"
                " modular-arithmetic grokking.",
                [h[1, -1], h[2, -1]],
                ["training accuracy", "held-out accuracy"],
            ),
        ],
        source="MNIST official train/test splits",
        settings={
            "steps": 3000,
            "train_images": 600,
            "test_images": 1000,
            "resolution": "7x7 area average",
        },
        dataset="MNIST",
        dataset_sha256=mnist.SHA256,
        dataset_provenance={"train": train_meta, "test": test_meta},
        weights_artifacts=[artifact],
        limitations=[
            "May show no grokking; report observed curve rather than manufacture a transition."
        ],
    )


def _scaling():
    from . import tinylm as lm

    sizes = [8, 12, 16, 24, 32]
    counts = [24, 48, 96, 192, 384, 768]
    seq = _text_windows(768, 12)
    test = _text_windows(128, 12, "test", SEED + 1)
    records = []
    artifacts = []
    for width in sizes:
        c = lm.Config(vocab=_corpus()["vocab"], width=width, heads=4, kv_heads=4, ff=2 * width)
        for count in counts:
            run = lm.train(
                c,
                seq[:count, :-1],
                seq[:count, 1:],
                steps=180,
                seed=SEED,
                name="real-scaling-%d-%d-v8" % (width, count),
            )
            parameters = sum(v.size for v in run["P"].values())
            loss = lm.loss(run["P"], test[:, :-1], test[:, 1:], c)
            records.append([parameters, count, 6 * parameters * 180 * min(count, 24) * 12, loss])
            artifacts.append(_save_weights("scaling-%d-%d" % (width, count), run["P"], steps=180))
    r = np.asarray(records)
    order = np.argsort(r[:, 2])
    lx = np.log(r[order, 2])
    ly = np.log(r[order, 3])
    fit = np.polyfit(lx, ly, 1)
    return _study(
        "Scaling: thirty real-text training runs",
        "TinyShakespeare · NumPy transformer size/data sweep",
        [
            _matrix(
                "Measured held-out loss",
                "Rows vary model width; columns vary the number of distinct training excerpts.",
                r[:, 3].reshape(5, 6),
            ),
            _panel(
                "Compute estimate versus held-out loss",
                "Horizontal coordinate is log10 estimated training FLOPs; classes identify width.",
                "points",
                xy=np.c_[np.log10(r[:, 2]), r[:, 3]],
                classes=np.repeat(np.arange(5), 6),
            ),
            _curve(
                "Descriptive log-loss fit",
                "No entropy floor is invented for real text; this is log total loss versus log"
                " compute.",
                [ly, np.polyval(fit, lx)],
                ["log measured loss", "least-squares fit"],
                x=lx,
                notes=("This small sweep is not a universal scaling law.",),
            ),
            _curve(
                "Best observed loss by budget",
                "Only measured runs define the frontier; compute is estimated, not timed.",
                np.minimum.accumulate(r[order, 3]),
                ["observed frontier"],
                x=r[order, 2],
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"runs": 30, "steps_each": 180},
        **_text_meta(weights_artifacts=artifacts, evaluation="128 held-out excerpts"),
    )


def _mixing():
    from . import tinylm as lm

    A = _domain_windows(0)
    B = _domain_windows(1)
    ta = _domain_windows(0, "test", 96)
    tb = _domain_windows(1, "test", 96)
    c = lm.Config(vocab=_corpus()["vocab"], width=24)
    curves = []
    artifacts = []
    for mode in ("90/10", "50/50", "curriculum"):
        rng = np.random.default_rng(SEED)
        P = lm.init(c, SEED)
        opt = Adam(P, 0.003)
        trace = []
        for step in range(321):
            if step % 8 == 0:
                trace.append(
                    [lm.loss(P, ta[:, :-1], ta[:, 1:], c), lm.loss(P, tb[:, :-1], tb[:, 1:], c)]
                )
            if step == 320:
                break
            ratio = 0.9 if mode == "90/10" else 0.5 if mode == "50/50" else 0.9 - 0.8 * step / 320
            n = int(round(24 * ratio))
            seq = np.r_[A[rng.integers(len(A), size=n)], B[rng.integers(len(B), size=24 - n)]]
            _, g = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], c)
            opt.step(P, g)
        curves.append(np.asarray(trace).T)
        artifacts.append(_save_weights("mixing-" + mode.replace("/", "-"), P, steps=320))
    labels = ["90/10", "50/50", "curriculum"]
    return _study(
        "Mixing two real-text regions",
        "TinyShakespeare · each source region has its own held-out tail",
        [
            _matrix(
                "Collected excerpts from both regions",
                "Both rows encode original Shakespeare characters, not generated transition rules.",
                np.array([A[0], B[0]]),
            ),
            _curve(
                "Region A held-out loss",
                "Mixture schedules change exposure to original source passages.",
                [v[0] for v in curves],
                labels,
                x=np.arange(41) * 8,
            ),
            _curve(
                "Region B held-out loss",
                "All schedules use the same untouched held-out region B excerpts.",
                [v[1] for v in curves],
                labels,
                x=np.arange(41) * 8,
            ),
            _bars(
                "Equal-weight held-out mean",
                "The evaluation weights both regions equally regardless of the training mixture.",
                [v[:, -1].mean() for v in curves],
                labels,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 320},
        **_text_meta(
            weights_artifacts=artifacts,
            evaluation="Two source halves, each contiguous80/20 train/test",
        ),
    )


def _tokenizers():
    from . import tinylm as lm

    train = _corpus()["train"][:18000]
    test = _corpus()["test"][:4000]
    length = []
    table = []
    losses = []
    curves = []
    artifacts = []
    names = ["characters", "BPE16", "BPE48", "BPE96", "bytes"]
    for merges in (0, 16, 48, 96, -1):
        if merges < 0:
            tr = list(train.encode())
            te = list(test.encode())
        else:
            rules = toycorpus.bpe_train(train, merges)
            tr = toycorpus.bpe_apply(train, rules)
            te = toycorpus.bpe_apply(test, rules)
        vocabulary = sorted(set(tr))
        mapping = {v: i + 1 for i, v in enumerate(vocabulary)}
        tr = np.asarray([mapping[v] for v in tr])
        te = np.asarray([mapping.get(v, 0) for v in te])
        starts = np.arange(0, len(tr) - 32, 32)
        x = np.asarray([tr[i : i + 32] for i in starts])
        y = np.asarray([tr[i + 1 : i + 33] for i in starts])
        cfg = lm.Config(vocab=len(vocabulary) + 1, width=16, heads=4, kv_heads=4, ff=32, max_len=32)
        run = lm.train(cfg, x, y, steps=220, seed=SEED, name="real-tokenizer-%d-v8" % merges)
        total = 0.0
        for i in range(0, len(te) - 1, 32):
            xx = te[i : min(i + 32, len(te) - 1)][None]
            yy = te[i + 1 : i + 1 + xx.shape[1]][None]
            total += nll(lm.forward(run["P"], xx, cfg), yy).sum()
        length.append(len(te))
        table.append((len(vocabulary) + 1) * 16)
        losses.append(total / len(test))
        curves.append(run["loss"])
        artifacts.append(
            _save_weights(
                "tokenizer-%d" % merges,
                run["P"],
                history={"loss": run["loss"]},
                steps=220,
                unknown_tokens=int((te == 0).sum()),
            )
        )
    return _study(
        "Tokenizers trained on real text",
        "TinyShakespeare · training-only vocabularies; ID0 handles unknown symbols",
        [
            _bars(
                "Same held-out passage, different tokens",
                "BPE merges are fitted only on the training excerpt.",
                length,
                names,
            ),
            _bars(
                "Embedding scalar count",
                "Vocabulary including unknown multiplied by width16; an exact parameter count.",
                table,
                names,
            ),
            _bars(
                "Held-out nats per original character",
                "Every model scores all but the first held-out token; divide by the same original"
                " character count.",
                losses,
                names,
            ),
            _curve(
                "Measured neural training losses",
                "Losses here are per token and are not directly comparable across tokenizers.",
                curves,
                names,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 220, "training_characters": len(train), "test_characters": len(test)},
        **_text_meta(weights_artifacts=artifacts),
    )


def _batching():
    import time

    rng = np.random.default_rng(SEED)
    weight = rng.normal(size=(384, 384)).astype(np.float32)
    batches = [1, 2, 4, 8, 16, 32, 64]
    secs = []
    for batch in batches:
        x = rng.normal(size=(batch, 384)).astype(np.float32)
        for _ in range(3):
            x @ weight
        start = time.perf_counter()
        for _ in range(120):
            x @ weight
        secs.append((time.perf_counter() - start) / 120)
    durations = np.array([7, 4, 9, 3, 6, 5, 8, 2, 7, 4, 6, 3])
    slots = np.full((4, 24), -1.0)
    queue = list(enumerate(durations))
    active = [None] * 4
    for t in range(24):
        for slot in range(4):
            if active[slot] is None and queue:
                active[slot] = list(queue.pop(0))
            if active[slot] is not None:
                idx, remaining = active[slot]
                slots[slot, t] = idx
                active[slot][1] -= 1
                if remaining == 1:
                    active[slot] = None
    intensity = np.array([2 * b * 384 * 384 / (4 * (384 * 384 + 2 * b * 384)) for b in batches])
    return _study(
        "Batching and throughput",
        "Measured NumPy projection timings · simulated serving slots",
        [
            _curve(
                "Rows processed per second",
                "The same 384 × 384 float32 projection is measured at each batch size.",
                np.array(batches) / np.array(secs),
                ["rows per second"],
                x=batches,
            ),
            _curve(
                "Time per matrix projection",
                "Each point averages 120 warmed matrix multiplies on this host.",
                secs,
                ["seconds per batch"],
                x=batches,
            ),
            _curve(
                "Arithmetic intensity",
                "FLOPs count multiply and add; bytes count weights, inputs and outputs.",
                intensity,
                ["FLOPs per byte"],
                x=batches,
            ),
            _matrix(
                "Continuous batching slots",
                "A completed request releases its slot immediately for the next queued request.",
                slots,
                notes=("Rows: four slots; columns: decode steps",),
            ),
        ],
        source="Local NumPy timing; exact request simulation",
        settings={"matrix": 384, "repeats": 120},
        limitations=["Timing is a projection microbenchmark, not an inference-server benchmark"],
    )


def _instruction():
    from . import tinylm as lm

    c = lm.Config(vocab=_corpus()["vocab"], width=24)
    raw = _text_windows(256, 16)
    base = lm.train(
        c, raw[:, :-1], raw[:, 1:], steps=180, seed=SEED, name="real-instruction-base-v8"
    )
    prefix = _encode_chars("Next:")
    separator = _encode_chars("\n")

    def examples(split, n):
        seq = _text_windows(n, 11, split, SEED + (split != "train"))
        formatted = np.c_[
            np.tile(prefix, (n, 1)), seq[:, :8], np.tile(separator, (n, 1)), seq[:, 8:]
        ]
        x, y = formatted[:, :-1], formatted[:, 1:]
        mask = np.zeros_like(y, float)
        mask[:, -4:] = 1
        return x, y, mask

    x, y, mask = examples("train", 256)
    xt, yt, _mt = examples("test", 96)
    runs = []
    artifacts = []
    for masked in (True, False):
        rng = np.random.default_rng(SEED)
        P = {k: v.copy() for k, v in base["P"].items()}
        opt = Adam(P, 0.003)
        trace = []
        for step in range(401):
            if step % 10 == 0:
                z = lm.forward(P, xt, c)
                trace.append(
                    [nll(z, yt)[:, -4:].mean(), (z[:, -4:].argmax(-1) == yt[:, -4:]).mean()]
                )
            if step == 400:
                break
            idx = rng.choice(len(x), 24, False)
            _, g = lm.loss_grads(P, x[idx], y[idx], c, weights=mask[idx] if masked else None)
            opt.step(P, g)
        runs.append(np.asarray(trace).T)
        artifacts.append(
            _save_weights("instruction-%s" % masked, P, history={"heldout": runs[-1]}, steps=400)
        )
    return _study(
        "Instruction-style continuation tuning",
        "Real Shakespeare continuations · automatically constructed 'Next:' prompts",
        [
            _matrix(
                "Observed text supplies every answer",
                "The fixed instruction wraps a real prefix; answer characters come from the"
                " original passage.",
                np.c_[x[:6], y[:6, -1:]],
            ),
            _matrix(
                "Answer-only supervision mask",
                "The mask selects four recorded continuation characters, not invented answers.",
                mask[:6],
            ),
            _curve(
                "Held-out answer loss",
                "Both fine-tunes start from the same real-text base and use identical minibatch"
                " indices.",
                [r[0] for r in runs],
                ["answer-only loss", "all-token loss"],
                x=np.arange(41) * 10,
            ),
            _curve(
                "Held-out answer-character accuracy",
                "This is automatic text-continuation supervision, not a human instruction-following"
                " benchmark.",
                [r[1] for r in runs],
                ["answer-only loss", "all-token loss"],
                x=np.arange(41) * 10,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"base_steps": 180, "adaptation_steps": 400, "answer_characters": 4},
        **_text_meta(weights_artifacts=artifacts),
    )


def adaptation_loss(P, extra, x, y, config, method, gradients=False):
    """One-layer S1 with learned input prompts or genuine per-layer KV prefixes."""
    from . import tinylm as lm

    params = {k: lm.Var(v) for k, v in P.items()}
    adapt = {k: lm.Var(v) for k, v in extra.items()}

    def cat(a, b, axis):
        split = a.v.shape[axis]

        def first(g):
            idx = [slice(None)] * g.ndim
            idx[axis] = slice(0, split)
            return g[tuple(idx)]

        def second(g):
            idx = [slice(None)] * g.ndim
            idx[axis] = slice(split, None)
            return g[tuple(idx)]

        return lm.Var(np.concatenate([a.v, b.v], axis), ((a, first), (b, second)))

    batch, length = x.shape
    h = params["E"][x]
    if method == "soft prompt":
        prompt = adapt["prompt"] + np.zeros((batch, 2, config.width))
        h = cat(prompt, h, 1)
    n = h.v.shape[1]
    h = h + params["pos"][np.arange(n)]
    z = lm._norm(h, params, "l0.n1", config)
    heads, dh = config.heads, config.width // config.heads
    q = (z @ params["l0.Q"]).reshape(batch, n, heads, dh).transpose(0, 2, 1, 3)
    k = (z @ params["l0.K"]).reshape(batch, n, heads, dh).transpose(0, 2, 1, 3)
    v = (z @ params["l0.V"]).reshape(batch, n, heads, dh).transpose(0, 2, 1, 3)
    mask = np.tril(np.ones((n, n), bool))
    if method == "KV prefix":
        pk = adapt["prefix_k"] + np.zeros((batch, heads, 2, dh))
        pv = adapt["prefix_v"] + np.zeros((batch, heads, 2, dh))
        k = cat(pk, k, 2)
        v = cat(pv, v, 2)
        mask = np.c_[np.ones((n, 2), bool), mask]
    a = lm.softmax((q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh) + np.where(mask, 0.0, -1e9))
    out = (a @ v).transpose(0, 2, 1, 3).reshape(batch, n, config.width)
    h = h + out @ params["l0.O"]
    z = lm._norm(h, params, "l0.n2", config)
    h = h + lm.gelu(z @ params["l0.W1"]) @ params["l0.W2"]
    h = lm._norm(h, params, "final", config)
    U = params["U"]
    if method == "LoRA":
        U = U + adapt["A"] @ adapt["B"]
    logits = h @ U
    if method == "soft prompt":
        logits = logits[:, 2:]
    shift = logits - logits.v.max(-1, keepdims=True)
    logp = shift - shift.exp().sum(-1, True).log()
    value = -logp[np.arange(batch)[:, None], np.arange(length), y].mean()
    if not gradients:
        return float(value.v)
    value.backward()
    gp = {k: np.zeros_like(v.v) if v.g is None else v.g for k, v in params.items()}
    ga = {k: np.zeros_like(v.v) if v.g is None else v.g for k, v in adapt.items()}
    return float(value.v), gp, ga


def _peft():
    from . import tinylm as lm

    source = _domain_windows(0, length=12)
    target = _domain_windows(1, length=12)
    test = _domain_windows(1, "test", 96, 12)
    c = lm.Config(vocab=_corpus()["vocab"], width=24)
    base = lm.train(
        c, source[:, :-1], source[:, 1:], steps=180, seed=SEED, name="real-peft-base-v8"
    )
    x, y = target[:, :-1], target[:, 1:]
    xt, yt = test[:, :-1], test[:, 1:]
    methods = ["soft prompt", "KV prefix", "LoRA", "full tuning"]
    traces = []
    sizes = []
    artifacts = []
    for method in methods:
        rng = np.random.default_rng(SEED)
        P = {k: v.copy() for k, v in base["P"].items()}
        extra = {}
        if method == "soft prompt":
            extra = {"prompt": rng.normal(0, 0.02, (1, 2, 24))}
        elif method == "KV prefix":
            extra = {
                "prefix_k": rng.normal(0, 0.02, (1, 4, 2, 6)),
                "prefix_v": rng.normal(0, 0.02, (1, 4, 2, 6)),
            }
        elif method == "LoRA":
            extra = {"A": rng.normal(0, 0.03, (24, 2)), "B": np.zeros((2, c.vocab))}
        trainable = P if method == "full tuning" else extra
        opt = Adam(trainable, 0.003)
        trace = []
        for step in range(221):
            if step % 10 == 0:
                trace.append(adaptation_loss(P, extra, xt, yt, c, method))
            if step == 220:
                break
            ix = rng.choice(len(x), 24, False)
            _, gp, ga = adaptation_loss(P, extra, x[ix], y[ix], c, method, True)
            opt.step(trainable, gp if method == "full tuning" else ga)
        traces.append(trace)
        sizes.append(sum(v.size for v in trainable.values()))
        artifacts.append(
            _save_weights(
                "peft-" + method.replace(" ", "-"),
                dict(P, **{"adapter_" + k: v for k, v in extra.items()}),
                history={"heldout_loss": trace},
                steps=220,
                method=method,
            )
        )
    return _study(
        "Parameter-efficient adaptation to real text",
        "TinyShakespeare · change source region, preserve held-out passages",
        [
            _bars(
                "Actually trainable scalars",
                "Only the highlighted adapters or full model are optimized.",
                sizes,
                methods,
            ),
            _curve(
                "Held-out next-character loss",
                "Every method adapts from the same real-text base model.",
                traces,
                methods,
                x=np.arange(23) * 10,
            ),
            _bars(
                "Final held-out loss",
                "A smaller adapter is not assumed to match full tuning.",
                [v[-1] for v in traces],
                methods,
            ),
            _bars(
                "Fraction of full model trained",
                "Soft prompt and KV prefix learn vectors; this LoRA modifies the output"
                " projection.",
                np.asarray(sizes) / sizes[-1],
                methods,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 220, "rank": 2},
        **_text_meta(weights_artifacts=artifacts, evaluation="domain B held-out contiguous tail"),
    )


def _merging():
    from . import tinylm as lm

    vocab = _corpus()["vocab"]
    c = lm.Config(vocab=vocab + 2, width=24)
    train = []
    test = []
    for domain in (0, 1):
        for split, out, n in (("train", train, 192), ("test", test, 96)):
            seq = _domain_windows(domain, split, n, 12)
            out.append((np.c_[np.full(n, vocab + domain), seq[:, :-1]], seq))
    base_run = lm.train(
        c,
        np.r_[train[0][0], train[1][0]],
        np.r_[train[0][1], train[1][1]],
        steps=100,
        seed=SEED,
        name="real-merging-base-v8",
    )
    base = base_run["P"]
    fine = []
    artifacts = []
    for i, (x, y) in enumerate(train):
        rng = np.random.default_rng(SEED)
        P = {k: v.copy() for k, v in base.items()}
        opt = Adam(P, 0.003)
        for _ in range(300):
            ix = rng.choice(len(x), 24, False)
            _, g = lm.loss_grads(P, x[ix], y[ix], c)
            opt.step(P, g)
        fine.append(P)
        artifacts.append(_save_weights("merge-domain-%d" % i, P, steps=300))
    models = [base, *fine]
    for a, b in ((0.5, 0.5), (1.0, 1.0), (1.0, -1.0)):
        models.append(
            {k: base[k] + a * (fine[0][k] - base[k]) + b * (fine[1][k] - base[k]) for k in base}
        )
    scores = np.asarray([[lm.loss(P, x, y, c) for x, y in test] for P in models]).T
    names = ["base", "region A", "region B", "average", "sum", "A minus B"]
    return _study(
        "Merging real-text adaptations",
        "TinyShakespeare · independently fine-tuned source regions",
        [
            _matrix(
                "Two actual source regions",
                "A task ID marks the corpus region; all prediction targets are real recorded"
                " characters.",
                np.r_[train[0][0][:4], train[1][0][:4]],
            ),
            _bars(
                "Learned displacement norms",
                "Each task vector subtracts the shared pretrained weights.",
                [np.sqrt(sum(((p[k] - base[k]) ** 2).sum() for k in base)) for p in fine],
                ["region A", "region B"],
            ),
            _bars(
                "Region A held-out loss",
                "Weight arithmetic is tested on text never used for these fine-tunes.",
                scores[0],
                names,
            ),
            _bars(
                "Region B held-out loss",
                "Interference is measured; successful skill addition is not assumed.",
                scores[1],
                names,
            ),
        ],
        source="Collected TinyShakespeare text",
        settings={"steps": 300},
        **_text_meta(
            weights_artifacts=artifacts,
            evaluation="separate contiguous80/20 split in each source region",
        ),
    )


def _induction():
    from . import tinylm as lm

    seq = _text_windows(384, 31)
    test = _text_windows(64, 31, "test", SEED + 1)
    c = lm.Config(vocab=_corpus()["vocab"], width=32, layers=2, ff=48)
    P = lm.init(c, SEED)
    for layer in range(2):
        P["l%d.W2" % layer][:] = 0
    opt = Adam(P, 0.003)
    rng = np.random.default_rng(SEED)
    losses = []
    for _ in range(650):
        ix = rng.choice(len(seq), 24, False)
        lo, g = lm.loss_grads(P, seq[ix, :-1], seq[ix, 1:], c)
        for layer in range(2):
            g["l%d.W1" % layer][:] = 0
            g["l%d.W2" % layer][:] = 0
        opt.step(P, g)
        losses.append(lo)
    _, rec = lm.forward(P, test[:8, :-1], c, keep=True)
    attention = np.asarray(rec["attention"])

    def successor(attn, ids):
        rows = []
        for t in range(2, len(ids)):
            js = [j + 1 for j in range(t - 1) if ids[j] == ids[t]]
            if js:
                rows.append(attn[..., t, js].sum(-1))
        return np.mean(rows, axis=0) if rows else np.zeros(attn.shape[:-2])

    scores = np.mean([successor(attention[:, i], test[i, :-1]) for i in range(8)], axis=0)
    text = _corpus()["test"][:220]
    ids = _tok().encode(text)
    gpt = _forward(text, record=True)
    mass = successor(gpt["attn"], ids)
    held = lm.loss(P, test[:, :-1], test[:, 1:], c)
    artifact = _save_weights(
        "induction-real-text",
        P,
        history={"training_loss": losses, "successor_mass": scores},
        steps=650,
    )
    return _study(
        "Searching for successor attention in real text",
        "Attention-only character transformer · natural repeated contexts",
        [
            _matrix(
                "Untouched source excerpts",
                "Characters are sampled from held-out Shakespeare; no sequence is duplicated to"
                " manufacture an induction task.",
                test[:8],
            ),
            _curve(
                "Actual next-character training",
                "Two attention-only blocks optimize ordinary collected-text prediction.",
                losses,
                ["training loss"],
                notes=("Held-out nats / character: %.3f" % held,),
            ),
            _matrix(
                "Repeated-character successor attention",
                "For each eligible query, sum attention on successors of earlier matching"
                " characters, then average.",
                scores,
            ),
            _matrix(
                "GPT-2 natural-text probe",
                "Natural token repeats are evaluated separately; an attention pattern alone does"
                " not prove induction.",
                mass,
            ),
        ],
        source="Collected TinyShakespeare and pretrained GPT-2",
        settings={"steps": 650, "layers": 2},
        **_text_meta(
            weights_artifacts=[artifact],
            heldout_loss=held,
            limitations=["Natural repetition need not produce a strong induction head."],
        ),
    )


def _input_gradient(params, x, target):
    _acts, zs, _p = __import__("voxanim.mnist", fromlist=["_forward"])._forward(params, x)
    delta = np.zeros((len(x), 10))
    delta[:, target] = 1
    for k in range(len(params) - 1, -1, -1):
        delta = delta @ params[k][0].T
        if k > 0:
            delta *= zs[k - 1] > 0
    return delta


def integrated_gradients(params, x, target, steps=256):
    """Midpoint integration of the target logit gradient from the black image."""
    grid = (np.arange(steps) + 0.5) / steps
    grad = _input_gradient(params, grid[:, None] * x[None], target).mean(0)
    return x * grad


def _saliency():
    from . import mnist
    from .netanim import div_colors

    params, _acc = mnist.train()
    _, _, test, y = mnist.load()
    x = test[0]
    target = int(y[0])
    gradient = _input_gradient(params, x[None], target)[0]
    ig = integrated_gradients(params, x, target, 512)
    _, z, _ = mnist._forward(params, np.array([np.zeros(784), x]))
    gap = float(z[-1][1, target] - z[-1][0, target])
    vmax = max(
        float(np.abs(gradient).max()),
        float(np.abs(gradient * x).max()),
        float(np.abs(ig).max()),
        1e-12,
    )

    def colors(v):
        return div_colors(np.asarray(v).ravel(), vmax).reshape(28, 28, 3)

    return _study(
        "Saliency and integrated gradients",
        "MNIST MLP · target logit attribution",
        [
            _panel(
                "The classified digit",
                "Attributions explain the selected class logit of the trained MNIST MLP.",
                "images",
                values=x.reshape(1, 28, 28),
                labels=["digit %d" % target],
            ),
            _panel(
                "Raw gradient and gradient × input",
                "The raw derivative is local; multiplying by input removes the black background.",
                "images",
                values=np.stack([colors(gradient), colors(gradient * x)]),
                labels=["gradient", "gradient × input"],
            ),
            _panel(
                "Integrated gradients",
                "Integrate along 512 midpoint samples from a black reference image.",
                "images",
                values=colors(ig)[None],
                labels=["integrated gradients"],
                notes=("Orange positive; blue negative; shared scale.",),
            ),
            _bars(
                "Completeness check",
                "Summed attributions approximate the input-minus-baseline target logit.",
                [gap, ig.sum(), abs(gap - ig.sum())],
                ["logit difference", "sum attribution", "absolute error"],
            ),
        ],
        source="MNIST and trained repository MLP",
        settings={"index": 0, "integration_steps": 512},
    )


def cnn_feature_grad(params, image, channel):
    """Mean post-ReLU conv-channel activation and its input gradient."""
    from . import mnist

    x = np.asarray(image).reshape(1, 28, 28)
    cols = mnist._im2col(x)
    pre = (cols @ params[0][0].reshape(9, 4) + params[0][1]).reshape(26, 26, 4)
    grad = np.zeros((28, 28))
    active = (pre[:, :, channel] > 0) / (26 * 26)
    for i in range(3):
        for j in range(3):
            grad[i : i + 26, j : j + 26] += active * params[0][0][i, j, channel]
    return float(np.maximum(pre[:, :, channel], 0).mean()), grad


def _features():
    from . import mnist

    params, _acc = mnist.cnn_train()
    rng = np.random.default_rng(SEED)
    initial = rng.uniform(0.2, 0.8, (28, 28))
    snapshots = []
    losses = []
    for regularize in (False, True):
        image = initial.copy()
        trace = []
        frames = []
        for step in range(121):
            offset = tuple(rng.integers(-2, 3, size=2)) if regularize else (0, 0)
            jittered = np.roll(image, offset, axis=(0, 1))
            _, grad = cnn_feature_grad(params, jittered, 0)
            grad = np.roll(grad, tuple(-v for v in offset), axis=(0, 1))
            score, _ = cnn_feature_grad(params, image, 0)
            trace.append(score)
            if step % 30 == 0:
                frames.append(image.copy())
            if step == 120:
                break
            if regularize:
                # Laplacian smoothness derivative and an L2 brightness penalty.
                lap = sum(np.roll(image, s, axis=a) for a in (0, 1) for s in (-1, 1)) - 4 * image
                grad += 0.01 * lap - 0.001 * image
            image = np.clip(image + 2 * grad, 0, 1)
            if regularize and step % 4 == 0:
                image = (
                    4 * image + sum(np.roll(image, s, axis=a) for a in (0, 1) for s in (-1, 1))
                ) / 8
        snapshots.append(frames)
        losses.append(trace)
    return _study(
        "Feature visualization",
        "MNIST CNN · optimize input for convolution channel zero",
        [
            _panel(
                "Start from noise",
                "Both runs begin with exactly the same seeded image.",
                "images",
                values=initial[None],
                labels=["initial image"],
            ),
            _panel(
                "Unregularized gradient ascent",
                "The image maximizes the measured mean activation of channel zero.",
                "images",
                values=np.asarray(snapshots[0][::2]),
                labels=["step 0", "step 60", "step 120"],
            ),
            _panel(
                "Smoothness-regularized ascent",
                "Jitter, blur and a smoothness penalty discourage high-frequency patterns.",
                "images",
                values=np.asarray(snapshots[1][::2]),
                labels=["step 0", "step 60", "step 120"],
            ),
            _curve(
                "Channel response",
                "The activation objective is measured after each image update.",
                losses,
                ["unregularized", "smoothed"],
                notes=("Objective: mean first-layer ReLU response",),
            ),
        ],
        source="MNIST and trained repository CNN",
        settings={"channel": 0, "steps": 120},
    )


BUILDERS = {
    "teacher-forcing": _teacher,
    "perplexity-explained": _perplexity,
    "packing-and-padding": _packing,
    "warmup-and-clipping": _warmup,
    "grokking": _grokking,
    "scaling-laws": _scaling,
    "curriculum-and-data-mixing": _mixing,
    "tokenizer-effects": _tokenizers,
    "beam-search": _beam,
    "speculative-decoding": _speculative,
    "repetition-and-penalties": _penalties,
    "constrained-decoding": _constrained,
    "contrastive-decoding": _contrastive,
    "batching-and-throughput": _batching,
    "instruction-tuning": _instruction,
    "reward-model": _reward_model,
    "rlhf-ppo": _ppo,
    "dpo": _dpo,
    "reward-hacking": _reward_hacking,
    "prompt-and-prefix-tuning": _peft,
    "model-merging": _merging,
    "knowledge-editing": _knowledge_edit,
    "superposition": _superposition,
    "sparse-autoencoder-features": _sae,
    "linear-probes": _probes,
    "activation-patching": _patching,
    "induction-heads": _induction,
    "head-ablation": _ablation,
    "saliency-and-integrated-gradients": _saliency,
    "feature-visualisation": _features,
}


def make(name):
    """Build one complete cached measured study (optional gpt2- prefix accepted)."""
    name = name.removeprefix("gpt2-")
    return _cache(name, BUILDERS[name]).validate()

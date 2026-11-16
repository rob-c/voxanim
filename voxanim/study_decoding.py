"""Redesigned experiments for four of the decoding animations (30-33).

The first versions (:mod:`voxanim.next_language`) each rested on one authored prompt, and
speculative decoding could not show its point: its draft was GPT-2's first four layers read
through the final unembedding, which agrees with GPT-2 on about an eighth of the probability
and costs half a GPT-2 call a guess, so twelve guesses kept two and the run was slower.

Here speculative decoding first *measures* candidate drafts on held-out text - early exits
at eight depths, the same exits with a trained linear read-out, and n-gram tables counted
from the training text - and picks one by a rule stated before any decoding is run; the pick
and the first design are then both run on several prompts and seeds. The other three keep
their worked example and add the same measurement over twelve held-out passages.

Every function ending ``_data`` is the experiment (seeded, plain numpy, cached); the
function registered in :data:`STUDIES` wraps its numbers as a four-panel
:class:`~voxanim.study.Study`. :mod:`voxanim.mech_decoding` draws from both.

GPT-2 is read here by :class:`Stream`, an incremental (key-value cached) forward pass:
:func:`voxanim.gpt2_hooks.forward` spends 0.15 s of every call on its unembedding (numpy
multiplies by the transposed embedding table slowly), which a thousand one-token steps
cannot afford. The tests hold the two to the same logits.
"""

from __future__ import annotations

import itertools
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from . import gpt2, realdata
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
SEED = 23
V = gpt2.N_VOCAB


def _settings(name, **extra):
    return dict(version=VERSION, seed=SEED, name=name, gpt2_revision=gpt2.REVISION,
                dataset_sha256=realdata.TEXT_SHA256, **extra)


def _cached(name, build, **extra):
    return cached_npz("study-decoding-" + name, _settings(name, **extra), build)


def _text_meta(**extra):
    return dict(
        seed=SEED, study_version=VERSION,
        dataset="Tiny Shakespeare / char-rnn", dataset_sha256=realdata.TEXT_SHA256,
        dataset_provenance={k: realdata.text_metadata(k) for k in ("train", "validation", "test")},
        **extra)


@lru_cache(maxsize=1)
def _tok():
    return gpt2.Tokenizer()


def show(token):
    """A GPT-2 token as a label: its text without the leading space that marks a new word."""
    s = _tok().decode([int(token)]).replace("\n", "↵").strip()
    return s if s else "␣"


# ---------------------------------------------------------------------- GPT-2, token by token
class Stream:
    """GPT-2's first ``n_layers`` blocks read incrementally: :meth:`feed` takes the next
    tokens and returns the residual stream after the last block for each of them, keeping
    every layer's keys and values so that earlier tokens are never read again. Setting
    :attr:`n` back forgets the tokens after it (a rejected draft, a new rollout from the
    same prompt)."""

    def __init__(self, n_layers=gpt2.N_LAYER, room=128):
        self.W, self.layers, self.n = gpt2.weights(), n_layers, 0
        self.k = np.zeros((n_layers, 12, room, 64), np.float32)
        self.v = np.zeros((n_layers, 12, room, 64), np.float32)

    def feed(self, ids, taps=()):
        """The stream after the last block for the new tokens ``ids`` (rows). With ``taps``
        (block counts), a dict instead: the stream after that many blocks."""
        W, n = self.W, self.n
        ids = np.asarray(ids, np.int64)
        m = len(ids)
        x = W["wte.weight"][ids] + W["wpe.weight"][n:n + m]
        hidden = np.arange(n + m)[None, :] > (n + np.arange(m))[:, None]
        out = {}
        for layer in range(self.layers):
            if layer in taps:
                out[layer] = x.copy()
            pre = "h.%d." % layer
            h = gpt2.layer_norm(x, W[pre + "ln_1.weight"], W[pre + "ln_1.bias"])
            qkv = h @ W[pre + "attn.c_attn.weight"] + W[pre + "attn.c_attn.bias"]
            q, k, v = (a.reshape(m, 12, 64).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=1))
            self.k[layer, :, n:n + m], self.v[layer, :, n:n + m] = k, v
            keys, values = self.k[layer, :, :n + m], self.v[layer, :, :n + m]
            score = np.where(hidden[None], -np.inf, q @ keys.transpose(0, 2, 1) / 8)
            a = gpt2.softmax(score).astype(x.dtype)
            x = x + (a @ values).transpose(1, 0, 2).reshape(m, 768) \
                @ W[pre + "attn.c_proj.weight"] + W[pre + "attn.c_proj.bias"]
            h = gpt2.layer_norm(x, W[pre + "ln_2.weight"], W[pre + "ln_2.bias"])
            x = x + gpt2._gelu(h @ W[pre + "mlp.c_fc.weight"] + W[pre + "mlp.c_fc.bias"]) \
                @ W[pre + "mlp.c_proj.weight"] + W[pre + "mlp.c_proj.bias"]
        self.n = n + m
        if taps:
            out[self.layers] = x
            return out
        return x


def unembed(x):
    """Logits from residual-stream rows: the final layer norm, then the embedding table."""
    W = gpt2.weights()
    h = gpt2.layer_norm(np.atleast_2d(x), W["ln_f.weight"], W["ln_f.bias"])
    return (W["wte.weight"] @ h.T).T          # not h @ wte.T: numpy is 30 times slower at that


def probs(x):
    """Next-token probabilities (float64, each row summing to 1) from residual rows."""
    z = unembed(x).astype(np.float64)
    z -= z.max(-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(-1, keepdims=True)
    return z


def _softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(-1, keepdims=True)


def heldout_prompts(count, length):
    """``count`` passages of ``length`` GPT-2 tokens from the *test* split of Tiny
    Shakespeare, evenly spread through it, each starting at the start of a line."""
    text = realdata.text("test")
    out = []
    for at in np.linspace(0, len(text) - 20 * length, count).astype(int):
        start = 0 if at == 0 else text.index("\n", at) + 1
        while text[start] == "\n":
            start += 1
        out.append(_tok().encode(text[start:start + 20 * length])[:length])
    return np.asarray(out)


@lru_cache(maxsize=2)
def _split_tokens(split):
    return np.asarray(_tok().encode(realdata.text(split)))


def _windows(ids, count, length, rng):
    starts = rng.choice(len(ids) - length, count, replace=False)
    return [ids[s:s + length] for s in np.sort(starts)]


# ---------------------------------------------------------------------- 30 speculative decoding
EXIT_LAYERS = (0, 1, 2, 3, 4, 6, 8, 10)
BLOCKS = (1, 2, 3, 4)                    # how many tokens a draft may guess per round
NGRAMS = ("unigram", "bigram", "trigram")
SPEC_PROMPTS, SPEC_PROMPT_TOKENS, SPEC_SEEDS, SPEC_NEW = 8, 24, 3, 24
FIT_WINDOWS, VAL_WINDOWS, WINDOW, SKIP = 96, 24, 64, 8
RIDGE = 1e-2
CONTROL = ("exit", 4, 3)                 # the first design: 4-layer exit, three guesses a round
MC_DRAWS = 200_000

# multiply-adds in GPT-2's weight matrices for one new token (attention over the cached
# keys, which grows with the context, is left out for every model alike)
MADDS_LAYER = 768 * 2304 + 768 * 768 + 2 * 768 * 3072
MADDS_UNEMBED = 768 * V
MADDS_READOUT = 768 * 768
MADDS_TARGET = gpt2.N_LAYER * MADDS_LAYER + MADDS_UNEMBED


def draft_cost(kind, layer=0):
    """What one call of a draft costs, as a fraction of one GPT-2 call: the multiply-adds
    of its weight matrices for one token. A table look-up costs nothing."""
    if kind in NGRAMS:
        return 0.0
    extra = MADDS_READOUT if kind == "tuned" else 0
    return (layer * MADDS_LAYER + extra + MADDS_UNEMBED) / MADDS_TARGET


def tokens_per_pass(alpha, block):
    """Expected tokens one GPT-2 pass yields when each of ``block`` guesses is kept with
    chance ``alpha`` and the first rejection ends the round: 1 + a + a^2 + ... + a^block."""
    return float(sum(alpha ** i for i in range(block + 1)))


def predicted_speedup(alpha, cost, block):
    """Time per token of plain decoding over that of speculative decoding, when a GPT-2
    pass takes one unit however many guesses it checks and a draft call takes ``cost``."""
    return tokens_per_pass(alpha, block) / (1.0 + block * cost)


class NGram:
    """Unigram, bigram and trigram tables counted from a token sequence, each smoothed
    into the shorter one by Witten-Bell's rule (the weight of the shorter context is the
    share of distinct continuations among the longer one's sightings)."""

    def __init__(self, ids):
        counts = np.bincount(ids, minlength=V).astype(float)
        self.uni = (counts + 0.1) / (counts.sum() + 0.1 * V)
        self.bi, self.tri = defaultdict(Counter), defaultdict(Counter)
        ids = [int(i) for i in ids]
        for a, b in itertools.pairwise(ids):
            self.bi[a][b] += 1
        for a, b, c in zip(ids, ids[1:], ids[2:]):
            self.tri[(a, b)][c] += 1

    @staticmethod
    def _blend(seen, lower):
        if not seen:
            return lower
        n, kinds = sum(seen.values()), len(seen)
        q = lower * (kinds / (n + kinds))
        q[np.fromiter(seen.keys(), np.int64)] += np.fromiter(seen.values(), float) / (n + kinds)
        return q

    def q(self, kind, context):
        """The table's next-token distribution after the tokens ``context``."""
        q = self.uni.copy()
        if kind != "unigram" and len(context) >= 1:
            q = self._blend(self.bi.get(int(context[-1])), q)
        if kind == "trigram" and len(context) >= 2:
            q = self._blend(self.tri.get((int(context[-2]), int(context[-1]))), q)
        return q


def fit_readouts(windows, layers=EXIT_LAYERS, ridge=RIDGE):
    """For each exit depth a linear map (768 + 1 -> 768) from the stream at that depth to
    GPT-2's final stream, by ridge least squares on the positions of ``windows``: a "tuned
    lens" fitted in closed form, so that the final layer norm and unembedding read an early
    exit as nearly as a linear map allows. Returns ``{layer: (769, 768) array}``."""
    xtx = {layer: np.zeros((769, 769)) for layer in layers}
    xty = {layer: np.zeros((769, 768)) for layer in layers}
    stream = Stream()
    for window in windows:
        stream.n = 0
        out = stream.feed(window, taps=layers)
        y = out[gpt2.N_LAYER].astype(np.float64)
        for layer in layers:
            x = np.c_[out[layer].astype(np.float64), np.ones(len(window))]
            xtx[layer] += x.T @ x
            xty[layer] += x.T @ y
    maps = {}
    for layer in layers:
        penalty = np.eye(769) * ridge * np.trace(xtx[layer][:768, :768]) / 768
        penalty[-1, -1] = 0.0
        maps[layer] = np.linalg.solve(xtx[layer] + penalty, xty[layer]).astype(np.float32)
    return maps


def read_exit(x, readout=None):
    """Next-token probabilities from an early exit's stream rows, through the trained
    ``readout`` (or, without one, straight through GPT-2's own final norm and unembedding)."""
    x = np.atleast_2d(x)
    if readout is not None:
        x = x @ readout[:-1] + readout[-1]
    return probs(x)


class Draft:
    """A draft model: :meth:`propose` gives its next-token distribution after the tokens
    ``seq`` (which must extend what it has read so far); :meth:`rewind` forgets what it
    read past the first ``n`` tokens."""

    def __init__(self, kind, layer=0, readout=None, ngram=None):
        self.kind, self.layer, self.readout, self.ngram = kind, layer, readout, ngram
        self.stream = Stream(layer) if kind in ("exit", "tuned") and layer else None
        self.cost = draft_cost(kind, layer)

    def propose(self, seq):
        if self.kind in NGRAMS:
            return self.ngram.q(self.kind, seq)
        if self.stream is None:                      # depth 0: the embedding alone
            W = gpt2.weights()
            x = W["wte.weight"][int(seq[-1])] + W["wpe.weight"][len(seq) - 1]
        else:
            x = self.stream.feed(seq[self.stream.n:])[-1]
        return read_exit(x, self.readout)[0]

    def rewind(self, n):
        if self.stream is not None:
            self.stream.n = min(self.stream.n, n)


def speculative_step(p, q, token, rng):
    """One checked guess: ``token`` was drawn from the draft's ``q``; GPT-2's distribution
    there is ``p``. Returns ``(kept, replacement)``: kept with chance ``min(1, p/q)``,
    otherwise a replacement drawn from the part of ``p`` above ``q``."""
    if rng.random() < min(1.0, p[token] / max(q[token], 1e-300)):
        return True, None
    residual = np.maximum(p - q, 0.0)
    return False, int(rng.choice(len(p), p=residual / residual.sum()))


def speculative_distribution(p, q):
    """The distribution of the token one checked guess leaves behind (kept or replaced):
    exactly ``p``."""
    kept = np.minimum(p, q)
    residual = np.maximum(p - q, 0.0)
    return kept + (1.0 - kept.sum()) * residual / max(residual.sum(), 1e-300)


def speculate(prompt, n_new, draft, block, rng, target=None, keep_rows=False):
    """``n_new`` tokens after ``prompt``, sampled from GPT-2 by speculative decoding: each
    round the draft guesses up to ``block`` tokens, one GPT-2 pass reads them all, guesses
    are kept in order until the first rejection (replaced by GPT-2's correction), and when
    all are kept GPT-2's pass also supplies the token after them. Returns the tokens, one
    record per round and the largest identity error (see :func:`speculative_distribution`).

    The pass computes the residual stream for every guess at once; a guess's row is
    unembedded only when its turn to be checked comes (in numpy the unembedding is a third
    of a call, and rows after a rejection are never needed). ``keep_rows`` keeps each
    round's stream rows in its record."""
    target = target or Stream()
    target.n = 0
    draft.rewind(0)
    seq, rounds, worst = [int(i) for i in prompt], [], 0.0
    while len(seq) - len(prompt) < n_new:
        size = min(block, n_new - (len(seq) - len(prompt)) - 1)
        base, guesses, qs = len(seq), [], []
        for _ in range(size):
            q = draft.propose(seq + guesses)
            guesses.append(int(rng.choice(V, p=q)))
            qs.append(q)
        # one pass: the tokens GPT-2 has not read yet, then every guess
        rows = target.feed((seq + guesses)[target.n:])[-(size + 1):]
        state, extra, kept, ps = [0] * size, None, 0, [0.0] * size
        for i, (token, q) in enumerate(zip(guesses, qs)):
            p = probs(rows[i:i + 1])[0]
            ps[i] = float(p[token])
            worst = max(worst, float(np.abs(speculative_distribution(p, q) - p).max()))
            good, extra = speculative_step(p, q, token, rng)
            state[i] = 1 if good else 2
            if not good:
                break
            kept += 1
        if extra is None:
            p = probs(rows[kept:kept + 1])[0]
            extra = int(rng.choice(V, p=p))
        rounds.append(dict(start=base - len(prompt), guesses=guesses, state=state,
                           q=[float(q[t]) for t, q in zip(guesses, qs)], p=ps,
                           extra=extra, extra_p=float(p[extra]), after_rejection=kept < size,
                           rows=rows if keep_rows else None))
        seq = seq[:base] + guesses[:kept] + [extra]
        target.n = base + kept
        draft.rewind(base + kept)
    return seq[len(prompt):], rounds, worst


def plain_sample(prompt, n_new, rng, target=None):
    """``n_new`` tokens sampled from GPT-2 one pass at a time."""
    target = target or Stream()
    target.n = 0
    seq = [int(i) for i in prompt]
    for _ in range(n_new):
        p = probs(target.feed(seq[target.n:])[-1:])[0]
        seq.append(int(rng.choice(V, p=p)))
    return seq[len(prompt):]


def speculative_data():
    """The whole speculative-decoding experiment.

    1. *Candidates, measured on held-out text.* For every position of ``VAL_WINDOWS``
       validation windows, the share of probability a draft and GPT-2 agree on,
       ``sum(min(p, q))`` - the chance a guess drawn from the draft is kept. Drafts: an
       early exit at each depth of :data:`EXIT_LAYERS` read through GPT-2's own unembedding
       (``alpha_exit``), the same exits through a linear read-out fitted on ``FIT_WINDOWS``
       training windows (``alpha_tuned``), and three n-gram tables counted from the training
       split (``alpha_ngram``).
    2. *The rule.* Each candidate and block size is scored by :func:`predicted_speedup`
       from its measured agreement and its :func:`draft_cost`; the best is run
       (``chosen``, ``chosen_block``).
    3. *The runs.* On :data:`SPEC_PROMPTS` test passages and :data:`SPEC_SEEDS` seeds:
       plain sampling, speculative decoding with the chosen draft, and with the first
       design's draft (:data:`CONTROL`). Arrays ``(arm, prompt, seed)`` with arm 0 the
       chosen draft and 1 the control: ``passes`` (GPT-2 calls), ``drafts`` (draft calls),
       ``checked`` / ``kept`` guesses, ``seconds``; ``plain_seconds`` ``(prompt, seed)``.
    4. *Exactness.* ``identity_error``: the largest gap, over every checked guess, between
       GPT-2's distribution and what keep-or-replace leaves. ``mc_*``: at each prompt's
       first new token, the total-variation distance from GPT-2's distribution of
       :data:`MC_DRAWS` tokens produced by keep-or-replace, beside that of as many drawn
       from GPT-2 directly (sampling noise alone).

    ``trace_*``: every round of the chosen draft's run on prompt 0, seed 0 (``trace_p`` is
    filled in afterwards for the guesses GPT-2 never checked, to show what they were worth).
    ``id_*``: the two models at the first guess that run saw rejected (round and slot
    ``id_spot``, after the tokens ``id_context``) - their probabilities of the tokens either
    ranks in its top four, and how often each came out of the simulated steps; ``overlap``
    the probability they share there.
    ``train_tokens``: the size of the text the tables were counted from. ``load_average``
    and ``cpus``: how busy the machine was when the seconds were measured."""
    def build():
        train, val = _split_tokens("train"), _split_tokens("validation")
        rng = np.random.default_rng(SEED)
        ngram = NGram(train)
        readouts = fit_readouts(_windows(train, FIT_WINDOWS, WINDOW, rng))
        # ---- 1 agreement on validation text
        agree = defaultdict(list)
        stream = Stream()
        positions = np.arange(SKIP, WINDOW - 1)
        for window in _windows(val, VAL_WINDOWS, WINDOW, rng):
            stream.n = 0
            out = stream.feed(window, taps=EXIT_LAYERS)
            p = probs(out[gpt2.N_LAYER][positions])
            for layer in EXIT_LAYERS:
                agree["exit", layer] += list(np.minimum(p, read_exit(out[layer][positions])).sum(1))
                agree["tuned", layer] += list(np.minimum(
                    p, read_exit(out[layer][positions], readouts[layer])).sum(1))
            for i, at in enumerate(positions):
                for kind in NGRAMS:
                    agree[kind, 0].append(np.minimum(p[i], ngram.q(kind, window[:at + 1])).sum())
        # ---- 2 the rule
        cands = [("exit", layer) for layer in EXIT_LAYERS] + \
            [("tuned", layer) for layer in EXIT_LAYERS] + [(kind, 0) for kind in NGRAMS]
        alpha = np.asarray([np.mean(agree[c]) for c in cands])
        sem = np.asarray([np.std(agree[c]) / np.sqrt(len(agree[c])) for c in cands])
        cost = np.asarray([draft_cost(*c) for c in cands])
        speed = np.asarray([[predicted_speedup(a, c, b) for b in BLOCKS]
                            for a, c in zip(alpha, cost)])
        chosen, chosen_block = (int(i) for i in np.unravel_index(speed.argmax(), speed.shape))
        block = BLOCKS[chosen_block]

        def make(kind, layer):
            return Draft(kind, layer, readouts[layer] if kind == "tuned" else None, ngram)
        arms = [(make(*cands[chosen]), block), (make(*CONTROL[:2]), CONTROL[2])]
        # ---- 3 the runs
        prompts = heldout_prompts(SPEC_PROMPTS, SPEC_PROMPT_TOKENS)
        shape = (2, SPEC_PROMPTS, SPEC_SEEDS)
        passes, drafts, checked, kept = (np.zeros(shape, np.int64) for _ in range(4))
        seconds, plain_seconds = np.zeros(shape), np.zeros(shape[1:])
        outputs = np.zeros(shape + (SPEC_NEW,), np.int64)
        plain_out = np.zeros(shape[1:] + (SPEC_NEW,), np.int64)
        target, worst, trace = Stream(), 0.0, None
        plain_sample(prompts[0], 4, np.random.default_rng(0), target)        # warm the caches
        for pi, prompt in enumerate(prompts):
            for si in range(SPEC_SEEDS):
                start = time.perf_counter()
                plain_out[pi, si] = plain_sample(
                    prompt, SPEC_NEW, np.random.default_rng([SEED, pi, si, 9]), target)
                plain_seconds[pi, si] = time.perf_counter() - start
                for ai, (draft, size) in enumerate(arms):
                    start = time.perf_counter()
                    tokens, rounds, err = speculate(
                        prompt, SPEC_NEW, draft, size, np.random.default_rng([SEED, pi, si, ai]),
                        target, keep_rows=(ai, pi, si) == (0, 0, 0))
                    seconds[ai, pi, si] = time.perf_counter() - start
                    outputs[ai, pi, si] = tokens
                    passes[ai, pi, si] = len(rounds)
                    drafts[ai, pi, si] = sum(len(r["guesses"]) for r in rounds)
                    checked[ai, pi, si] = sum(sum(s > 0 for s in r["state"]) for r in rounds)
                    kept[ai, pi, si] = sum(sum(s == 1 for s in r["state"]) for r in rounds)
                    worst = max(worst, err)
                    if (ai, pi, si) == (0, 0, 0):
                        trace = rounds
        for record in trace:                       # what the unchecked guesses were worth
            for i, token in enumerate(record["guesses"]):
                if record["state"][i] == 0:
                    record["p"][i] = float(probs(record["rows"][i:i + 1])[0][token])
        # ---- 4 exactness, by simulation
        mc_rng = np.random.default_rng([SEED, 77])
        mc_spec, mc_direct = np.zeros(SPEC_PROMPTS), np.zeros(SPEC_PROMPTS)
        draft = arms[0][0]

        def simulate(context):
            """Both models' distributions after ``context``, and how often each token comes
            out of MC_DRAWS simulated keep-or-replace steps and of as many direct draws."""
            target.n = 0
            p = probs(target.feed(context)[-1:])[0]
            draft.rewind(0)
            q = draft.propose([int(i) for i in context])
            guess = mc_rng.choice(V, MC_DRAWS, p=q)
            keep = mc_rng.random(MC_DRAWS) < np.minimum(1.0, p[guess] / q[guess])
            residual = np.maximum(p - q, 0.0)
            final = np.where(keep, guess, mc_rng.choice(V, MC_DRAWS, p=residual / residual.sum()))
            direct = mc_rng.choice(V, MC_DRAWS, p=p)
            return p, q, np.bincount(final, minlength=V) / MC_DRAWS, \
                np.bincount(direct, minlength=V) / MC_DRAWS
        for pi, prompt in enumerate(prompts):
            p, q, made, direct = simulate(prompt)
            mc_spec[pi] = 0.5 * np.abs(made - p).sum()
            mc_direct[pi] = 0.5 * np.abs(direct - p).sum()
        # the position the drawing explains: the first guess the worked run saw rejected
        spot = next(((r, record["state"].index(2)) for r, record in enumerate(trace)
                     if 2 in record["state"]), (0, 0))
        before = trace[spot[0]]["start"] + spot[1]
        context = [int(i) for i in prompts[0]] + [int(i) for i in outputs[0, 0, 0][:before]]
        p, q, made, _ = simulate(context)
        both = list(np.argsort(-p)[:4])
        both += [t for t in np.argsort(-q)[:4] if t not in both]
        first = dict(id_tok=np.asarray(both), id_p=p[both], id_q=q[both], id_freq=made[both],
                     id_spot=np.asarray(spot), id_context=np.asarray(context[-4:]),
                     overlap=np.asarray(np.minimum(p, q).sum()))
        width = max(BLOCKS)

        def padded(key, fill, dtype):
            rows = np.full((len(trace), width), fill, dtype)
            for r, record in enumerate(trace):
                rows[r, :len(record[key])] = record[key]
            return rows
        return dict(
            layers=np.asarray(EXIT_LAYERS),
            cand_kind=np.asarray([NGRAMS.index(k) + 2 if k in NGRAMS else int(k == "tuned")
                                  for k, _ in cands]),
            cand_layer=np.asarray([layer for _, layer in cands]),
            alpha=alpha, alpha_sem=sem, cost=cost, speed=speed,
            chosen=np.asarray(chosen), chosen_block=np.asarray(block),
            positions=np.asarray(len(agree[cands[0]])), train_tokens=np.asarray(len(train)),
            load_average=np.asarray(os.getloadavg()[0]), cpus=np.asarray(os.cpu_count() or 0),
            prompts=prompts, outputs=outputs, plain_out=plain_out,
            passes=passes, drafts=drafts, checked=checked, kept=kept,
            seconds=seconds, plain_seconds=plain_seconds,
            identity_error=np.asarray(worst), mc_spec=mc_spec, mc_direct=mc_direct,
            trace_start=np.asarray([r["start"] for r in trace]),
            trace_prop=padded("guesses", -1, np.int64),
            trace_q=padded("q", 0.0, float), trace_p=padded("p", 0.0, float),
            trace_state=padded("state", -1, np.int64),
            trace_extra=np.asarray([r["extra"] for r in trace]),
            trace_extra_p=np.asarray([r["extra_p"] for r in trace]),
            trace_rejected=np.asarray([r["after_rejection"] for r in trace]),
            **first)
    return _cached("speculative", build, revision=4, prompts=SPEC_PROMPTS, seeds=SPEC_SEEDS,
                   new=SPEC_NEW, fit=FIT_WINDOWS, val=VAL_WINDOWS, ridge=RIDGE,
                   control=list(CONTROL), layers=list(EXIT_LAYERS), blocks=list(BLOCKS),
                   draws=MC_DRAWS)


KIND_NAMES = ("exit", "tuned") + NGRAMS


def candidate_name(kind, layer):
    """A candidate draft in words (``kind``: an index into :data:`KIND_NAMES`)."""
    kind = KIND_NAMES[int(kind)]
    if kind in NGRAMS:
        return "%s table" % kind
    if int(layer) == 0:
        return "embedding, %s" % ("trained read-out" if kind == "tuned" else "GPT-2's read-out")
    return "%d layer%s, %s" % (layer, "" if int(layer) == 1 else "s",
                               "trained read-out" if kind == "tuned" else "GPT-2's read-out")


def speculative_summary(d=None):
    """The run's headline numbers from :func:`speculative_data`: for each arm (0 the chosen
    draft, 1 the first design) the share of checked guesses kept, GPT-2 passes per token
    (overall and each seed's), draft calls per token and seconds, beside plain decoding's."""
    d = d or speculative_data()
    tokens = d["outputs"].shape[1] * d["outputs"].shape[3]              # per seed
    arms = []
    for a in range(2):
        per_seed = d["passes"][a].sum(0) / tokens
        arms.append(dict(
            kept=float(d["kept"][a].sum() / d["checked"][a].sum()),
            passes_per_token=float(d["passes"][a].sum() / (tokens * d["passes"].shape[2])),
            per_seed=[float(v) for v in per_seed],
            drafts_per_token=float(d["drafts"][a].sum() / (tokens * d["passes"].shape[2])),
            seconds=float(d["seconds"][a].sum()),
            per_seed_speedup=[float(v) for v in
                              d["plain_seconds"].sum(0) / d["seconds"][a].sum(0)]))
    return dict(arms=arms, plain_seconds=float(d["plain_seconds"].sum()),
                tokens=int(tokens * d["passes"].shape[2]))


def _speculative():
    d = speculative_data()
    s = speculative_summary(d)
    names = [candidate_name(k, layer) for k, layer in zip(d["cand_kind"], d["cand_layer"])]
    n = len(EXIT_LAYERS)
    chosen, block = int(d["chosen"]), int(d["chosen_block"])
    best = d["speed"].max(1)
    a, c = s["arms"]
    wall = [s["plain_seconds"] / a["seconds"], s["plain_seconds"] / c["seconds"]]
    return Study(
        title="Speculative decoding",
        subtitle="GPT-2 small · drafts measured on held-out Tiny Shakespeare, one chosen by rule",
        panels=(
            Panel("How often a draft agrees with GPT-2",
                  "The probability both models share, sum(min(p, q)), averaged over %d "
                  "held-out positions: the chance a guess is kept." % int(d["positions"]),
                  "curve",
                  dict(x=np.asarray(EXIT_LAYERS, float), ys=np.stack([d["alpha"][:n],
                                                                        d["alpha"][n:2 * n]]),
                       labels=["GPT-2's own read-out", "trained linear read-out"],
                       ngram=d["alpha"][2 * n:], ngram_labels=list(NGRAMS), sem=d["alpha_sem"]),
                  ("n-gram tables: " + ", ".join("%s %.2f" % (k, v) for k, v in
                                                 zip(NGRAMS, d["alpha"][2 * n:])),)),
            Panel("Predicted speed-up of every candidate",
                  "Tokens per GPT-2 pass over (1 + guesses x draft cost), at each candidate's "
                  "best block size; cost is the draft's multiply-adds over GPT-2's.",
                  "bars", dict(values=best, labels=names, cost=d["cost"], alpha=d["alpha"],
                               speed=d["speed"], blocks=np.asarray(BLOCKS)),
                  ("Chosen: %s, %d guesses a round (predicted %.2fx)" % (
                      names[chosen], block, d["speed"].max()),)),
            Panel("GPT-2 passes per token, measured",
                  "%d tokens: %d test passages x %d seeds x %d tokens. Plain decoding needs "
                  "one pass a token." % (s["tokens"], SPEC_PROMPTS, SPEC_SEEDS, SPEC_NEW),
                  "bars",
                  dict(values=np.asarray([1.0, a["passes_per_token"], c["passes_per_token"]]),
                       labels=["plain", "chosen draft", "first design"],
                       per_seed=np.asarray([a["per_seed"], c["per_seed"]]),
                       kept=np.asarray([a["kept"], c["kept"]]),
                       seconds=np.asarray([s["plain_seconds"], a["seconds"], c["seconds"]])),
                  ("Guesses kept: %.0f%% (chosen), %.0f%% (first design)" % (
                      100 * a["kept"], 100 * c["kept"]),
                   "Wall-clock in numpy, plain/speculative: %.2fx and %.2fx" % tuple(wall))),
            Panel("The output is still GPT-2's",
                  "Total-variation distance from GPT-2's distribution of %s tokens made by "
                  "keep-or-replace, and of as many drawn from GPT-2 directly." % (
                      f"{MC_DRAWS:,}"),
                  "curve",
                  dict(x=np.arange(1, SPEC_PROMPTS + 1), ys=np.stack([d["mc_spec"],
                                                                      d["mc_direct"]]),
                       labels=["speculative", "direct sampling"]),
                  ("Largest identity error over every checked guess: %.1e" % float(
                      d["identity_error"]),)),
        ),
        metadata=_text_meta(
            source="Pinned GPT-2; Tiny Shakespeare (tables and read-outs from the train split, "
                   "agreement on validation, decoding on test)",
            settings=dict(block=block, chosen=names[chosen], control=candidate_name(0, CONTROL[1]),
                          control_block=CONTROL[2], prompts=SPEC_PROMPTS, seeds=SPEC_SEEDS,
                          new_tokens=SPEC_NEW, fit_windows=FIT_WINDOWS, val_windows=VAL_WINDOWS,
                          window=WINDOW, ridge=RIDGE, temperature=1.0),
            selection_rule="largest predicted speed-up, tokens_per_pass(alpha, block) / (1 + "
                           "block * cost), over 19 candidate drafts and block sizes 1-4; alpha "
                           "from validation text, cost from multiply-add counts",
            results=dict(chosen=a, control=c, plain_seconds=s["plain_seconds"],
                         tokens=s["tokens"], predicted=float(d["speed"].max()),
                         predicted_control=float(predicted_speedup(
                             d["alpha"][EXIT_LAYERS.index(CONTROL[1])],
                             draft_cost(*CONTROL[:2]), CONTROL[2]))),
            timing_seconds=dict(baseline=s["plain_seconds"], speculative=a["seconds"],
                                control=c["seconds"], load_average=float(d["load_average"]),
                                cpus=int(d["cpus"])),
            designs_tried=[
                "first design (next_language): 4-layer exit through GPT-2's unembedding, 3 "
                "guesses, one prompt, 12 guesses - kept here as the control arm",
                "a cost model counting only transformer layers (which makes the embedding "
                "with a trained read-out free): replaced before any decoding run by "
                "multiply-add counts, because the unembedding is 31% of a GPT-2 call",
                "this design: 19 candidates measured on validation text, one chosen by the "
                "stated rule, run with the control on 8 test passages x 3 seeds"],
            limitations=[
                "Wall-clock is numpy on a CPU, where a pass over five tokens is not as cheap "
                "as a pass over one; the count of GPT-2 passes is the hardware-independent "
                "result",
                "The predicted speed-up assumes guesses are kept independently at the "
                "validation rate; sampled continuations drift from Shakespeare"]),
    )


# ---------------------------------------------------------------------- across held-out prompts
PROMPTS, PROMPT_TOKENS, ROLL = 12, 32, 24


def repeats_trigram(prompt, continuation):
    """True if the continuation completes a token trigram already present earlier in
    ``prompt + continuation``: the model has started copying itself."""
    seq = [int(i) for i in prompt] + [int(i) for i in continuation]
    seen = {tuple(seq[i:i + 3]) for i in range(len(prompt) - 2)}
    for end in range(len(prompt), len(seq)):
        gram = tuple(seq[end - 2:end + 1])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def distinct_fraction(continuation):
    """Distinct tokens over tokens."""
    return len({int(i) for i in continuation}) / len(continuation)


# ---------------------------------------------------------------------- 31 repetition
PENALTY_PROMPT = "I think that I think that I think that"
RULES = ("none", "repetition", "frequency", "trigram")
REPEAT_PENALTY, FREQUENCY_PENALTY = 1.3, 0.7


def reshape_logits(logits, seq, rule):
    """One of the three rules on next-token ``logits`` after the tokens ``seq``: the
    repetition penalty makes every used token's logit worse by a factor, the frequency
    penalty subtracts a step per use, the trigram rule forbids completing a trigram
    already written."""
    logits = np.asarray(logits, float)
    counts = np.bincount(seq, minlength=len(logits))
    if rule == "repetition":
        return np.where(counts > 0, np.where(logits > 0, logits / REPEAT_PENALTY,
                                             logits * REPEAT_PENALTY), logits)
    if rule == "frequency":
        return logits - FREQUENCY_PENALTY * counts
    if rule == "trigram":
        out = logits.copy()
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            if (a, b) == tuple(seq[-2:]):
                out[c] = -np.inf
        return out
    return logits


def rollout(prompt, steps, rule, stream):
    """GPT-2 continuing ``prompt`` greedily for ``steps`` tokens under ``rule``: the tokens,
    the (reshaped) probability of each, and the log-probability GPT-2 itself gave it."""
    stream.n = 0
    seq = [int(i) for i in prompt]
    tokens, shaped, own = [], [], []
    for _ in range(steps):
        logits = unembed(stream.feed(seq[stream.n:])[-1:])[0].astype(float)
        p = _softmax(reshape_logits(logits, seq, rule))
        tokens.append(int(p.argmax()))
        shaped.append(float(p.max()))
        own.append(float(np.log(_softmax(logits)[tokens[-1]])))
        seq.append(tokens[-1])
    return tokens, shaped, own


def penalty_data(steps=8):
    """The worked example and the measurement over held-out passages.

    Worked example (:data:`PENALTY_PROMPT`): ``top`` GPT-2's eight likeliest next tokens,
    ``logit`` their raw logits, ``prob`` their probability under each rule of :data:`RULES`,
    ``roll`` / ``roll_p`` the ``steps`` tokens GPT-2 then writes under each rule and the
    (reshaped) probability of each, ``banned`` how many tokens the trigram rule forbids.

    Across :data:`PROMPTS` test passages, :data:`ROLL` greedy tokens under each rule
    (arrays ``(rule, prompt)``): ``loops`` whether the continuation repeats a trigram
    (:func:`repeats_trigram`), ``distinct`` its share of distinct tokens, ``own`` the mean
    log-probability GPT-2 itself gave the tokens written; ``across_roll`` the tokens."""
    def build():
        stream = Stream()
        ids = _tok().encode(PENALTY_PROMPT)
        first = unembed(stream.feed(ids)[-1:])[0].astype(float)
        top = np.argsort(-_softmax(first))[:8]
        roll, roll_p = np.zeros((len(RULES), steps), np.int64), np.zeros((len(RULES), steps))
        for r, rule in enumerate(RULES):
            roll[r], roll_p[r], _ = rollout(ids, steps, rule, stream)
        prob = np.asarray([_softmax(reshape_logits(first, ids, rule))[top] for rule in RULES])
        prompts = heldout_prompts(PROMPTS, PROMPT_TOKENS)
        loops = np.zeros((len(RULES), PROMPTS), bool)
        distinct, own = np.zeros(loops.shape), np.zeros(loops.shape)
        across = np.zeros(loops.shape + (ROLL,), np.int64)
        for pi, prompt in enumerate(prompts):
            for r, rule in enumerate(RULES):
                tokens, _, logp = rollout(prompt, ROLL, rule, stream)
                across[r, pi] = tokens
                loops[r, pi] = repeats_trigram(prompt, tokens)
                distinct[r, pi] = distinct_fraction(tokens)
                own[r, pi] = np.mean(logp)
        return dict(top=top, logit=first[top], prob=prob, roll=roll, roll_p=roll_p,
                    banned=np.asarray(np.isneginf(reshape_logits(first, ids, "trigram")).sum()),
                    prompts=prompts, across_roll=across, loops=loops, distinct=distinct, own=own)
    return _cached("penalties", build, prompt=PENALTY_PROMPT, steps=steps, prompts=PROMPTS,
                   length=PROMPT_TOKENS, roll=ROLL, repeat=REPEAT_PENALTY,
                   frequency=FREQUENCY_PENALTY)


RULE_NAMES = ("no rule", "repetition %.1f" % REPEAT_PENALTY, "frequency %.1f" % FREQUENCY_PENALTY,
              "no repeated trigram")


def _penalties():
    d = penalty_data()
    return Study(
        title="Repetition and penalties",
        subtitle="GPT-2 small · one repeated prefix, then %d held-out passages" % PROMPTS,
        panels=(
            Panel("The next token under each rule",
                  "Rows: no rule, repetition penalty, frequency penalty, no repeated trigram. "
                  "Columns: GPT-2's eight likeliest tokens after the repeated prefix.",
                  "matrix", dict(values=d["prob"], labels=[show(t) for t in d["top"]],
                                 rules=list(RULE_NAMES)),
                  ("Banned tokens: %d" % int(d["banned"]),)),
            Panel("How often greedy decoding loops",
                  "Share of %d held-out passages whose %d-token greedy continuation repeats "
                  "a trigram already written." % (PROMPTS, ROLL),
                  "bars", dict(values=d["loops"].mean(1), labels=list(RULE_NAMES),
                               each=d["loops"].astype(float)),
                  ("The trigram rule makes a repeated trigram impossible by construction.",)),
            Panel("Distinct tokens in the continuation",
                  "Distinct tokens over tokens written, averaged over the passages.",
                  "bars", dict(values=d["distinct"].mean(1), labels=list(RULE_NAMES),
                               each=d["distinct"])),
            Panel("What the rule costs",
                  "Mean log-probability GPT-2 itself gave the tokens it was made to write.",
                  "bars", dict(values=d["own"].mean(1), labels=list(RULE_NAMES), each=d["own"])),
        ),
        metadata=_text_meta(
            source="Pinned GPT-2; prompts from the Tiny Shakespeare test split",
            settings=dict(repeat_penalty=REPEAT_PENALTY, frequency_penalty=FREQUENCY_PENALTY,
                          prompts=PROMPTS, prompt_tokens=PROMPT_TOKENS, new_tokens=ROLL),
            results=dict(loop_rate=[float(v) for v in d["loops"].mean(1)],
                         distinct=[float(v) for v in d["distinct"].mean(1)],
                         own_logp=[float(v) for v in d["own"].mean(1)]),
            limitations=["Greedy decoding only; a loop is a repeated token trigram, which the "
                         "trigram rule forbids by definition"]),
    )


# ---------------------------------------------------------------------- 32 constrained decoding
GRAMMAR_PROMPT = "Return only two digits, a comma, and two digits: "
GRAMMAR = ("digit", "digit", "comma", "digit", "digit")
FORM = re.compile(r"\d\d,\d\d")


def grammar_run(prompt, stream, digits, comma):
    """The grammar-constrained greedy run after ``prompt`` and the free one beside it.
    Per step: GPT-2's five likeliest tokens and their probabilities, its probability of
    each digit and of the comma, the mass the grammar lets through, the token emitted;
    then the five tokens GPT-2 writes with no grammar."""
    stream.n = 0
    seq = [int(i) for i in prompt]
    want_tok, want_p, digit_p, comma_p, mass, choice = [], [], [], [], [], []
    for kind in GRAMMAR:
        p = probs(stream.feed(seq[stream.n:])[-1:])[0]
        allowed = digits if kind == "digit" else [comma]
        top = np.argsort(-p)[:5]
        want_tok.append(top)
        want_p.append(p[top])
        digit_p.append(p[digits])
        comma_p.append(p[comma])
        mass.append(p[allowed].sum())
        choice.append(int(allowed[int(np.argmax(p[allowed]))]))
        seq.append(choice[-1])
    stream.n = 0
    free = [int(i) for i in prompt]
    for _ in GRAMMAR:
        free.append(int(unembed(stream.feed(free[stream.n:])[-1:])[0].argmax()))
    return dict(want_tok=np.asarray(want_tok), want_p=np.asarray(want_p),
                digit_p=np.asarray(digit_p), comma_p=np.asarray(comma_p),
                mass=np.asarray(mass), choice=np.asarray(choice),
                free=np.asarray(free[len(prompt):]))


def grammar_data():
    """The worked example (:data:`GRAMMAR_PROMPT`: the keys of :func:`grammar_run`, with
    ``digits`` / ``comma`` the eleven tokens the grammar ever allows) and the same run after
    :data:`PROMPTS` held-out passages, in two conditions (arrays ``(condition, prompt,
    ...)``): 0 the passage alone, where nothing invites digits, 1 the passage followed by
    the worked example's instruction. ``across_mass`` per step, ``across_choice`` the tokens
    emitted, ``across_free`` what GPT-2 writes unconstrained, ``across_valid`` /
    ``across_free_valid`` whether each text has the form two digits, comma, two digits."""
    def build():
        tok = _tok()
        digits = [tok.encode(str(d))[0] for d in range(10)]
        comma = tok.encode(",")[0]
        stream = Stream()
        out = grammar_run(tok.encode(GRAMMAR_PROMPT), stream, digits, comma)
        prompts = heldout_prompts(PROMPTS, PROMPT_TOKENS)
        suffix = tok.encode("\n\n" + GRAMMAR_PROMPT)
        shape = (2, PROMPTS, len(GRAMMAR))
        mass, choice, free = np.zeros(shape), np.zeros(shape, np.int64), np.zeros(shape, np.int64)
        valid, free_valid = np.zeros(shape[:2], bool), np.zeros(shape[:2], bool)
        for c in range(2):
            for pi, prompt in enumerate(prompts):
                run = grammar_run(list(prompt) + (suffix if c else []), stream, digits, comma)
                mass[c, pi], choice[c, pi], free[c, pi] = run["mass"], run["choice"], run["free"]
                valid[c, pi] = bool(FORM.fullmatch(tok.decode([int(i) for i in run["choice"]])))
                free_valid[c, pi] = bool(FORM.match(
                    tok.decode([int(i) for i in run["free"]]).strip()))
        return dict(out, digits=np.asarray(digits), comma=np.asarray(comma), prompts=prompts,
                    across_mass=mass, across_choice=choice, across_free=free,
                    across_valid=valid, across_free_valid=free_valid)
    return _cached("grammar", build, prompt=GRAMMAR_PROMPT, prompts=PROMPTS,
                   length=PROMPT_TOKENS)


def _constrained():
    d = grammar_data()
    tok = _tok()
    steps = ["digit 1", "digit 2", "comma", "digit 3", "digit 4"]
    asked = d["across_mass"][1]
    return Study(
        title="Constrained decoding",
        subtitle="GPT-2 small · grammar: DD,DD · one prompt, then %d held-out passages" % PROMPTS,
        panels=(
            Panel("The grammar mask", "Each state permits one ASCII digit, or the separator "
                  "comma.", "matrix",
                  dict(values=np.asarray([[1] * 10, [1] * 10, [0] * 10, [1] * 10, [1] * 10],
                                         float)),
                  ("Comma is the only token allowed at state 3.",)),
            Panel("Mass that survives", "The share of GPT-2's probability on tokens the grammar "
                  "allows, at each step of the worked prompt.", "bars",
                  dict(values=d["mass"], labels=steps)),
            Panel("Mass that survives, across passages",
                  "Rows: %d held-out passages, each followed by the instruction; columns: the "
                  "five steps." % PROMPTS, "matrix",
                  dict(values=asked, alone=d["across_mass"][0],
                       valid=d["across_valid"].astype(float),
                       free_valid=d["across_free_valid"].astype(float)),
                  ("Median kept per step: " + ", ".join("%.2g" % v for v in np.median(asked, 0)),
                   "Valid form: %d of %d with the grammar, %d of %d without" % (
                       d["across_valid"].sum(), d["across_valid"].size,
                       d["across_free_valid"].sum(), d["across_free_valid"].size))),
            Panel("A valid generated string", "Every emitted token obeys the current "
                  "finite-state grammar.", "bars",
                  dict(values=np.ones(5), labels=[tok.decode([int(i)]) for i in d["choice"]]),
                  ("Output: " + tok.decode([int(i) for i in d["choice"]]),)),
        ),
        metadata=_text_meta(
            source="Pinned GPT-2; prompts from the Tiny Shakespeare test split",
            settings=dict(grammar="DD,DD", prompts=PROMPTS, prompt_tokens=PROMPT_TOKENS),
            results=dict(
                median_mass_alone=[float(v) for v in np.median(d["across_mass"][0], 0)],
                median_mass_asked=[float(v) for v in np.median(asked, 0)],
                valid_with_grammar=int(d["across_valid"].sum()),
                valid_without=int(d["across_free_valid"].sum()), runs=int(d["across_valid"].size)),
            limitations=["The mask guarantees the form, not the content; single-digit tokens "
                         "only, so GPT-2's multi-digit tokens count as removed mass"]),
    )


# ---------------------------------------------------------------------- 33 contrastive decoding
CONTRAST_PROMPT = "A telescope is useful because it allows us to"
AMATEUR_LAYERS, ALPHA = 4, 0.1


def contrast(ep, ap):
    """Contrastive decoding's scores from the expert's and amateur's probabilities (rows):
    the plausibility mask (expert probability at least :data:`ALPHA` of its maximum), the
    log-probability difference, and the pick (largest difference among plausible tokens)."""
    plausible = ep >= ALPHA * ep.max(-1, keepdims=True)
    score = np.log(np.maximum(ep, 1e-300)) - np.log(np.maximum(ap, 1e-300))
    return plausible, score, np.where(plausible, score, -np.inf).argmax(-1)


def contrast_rollout(prompt, steps, expert, amateur, contrastive):
    """``steps`` tokens after ``prompt``: the expert's likeliest each time, or (with
    ``contrastive``) contrastive decoding's pick. Returns the tokens and the expert's
    log-probability of each."""
    expert.n = amateur.n = 0
    seq = [int(i) for i in prompt]
    tokens, logp = [], []
    for _ in range(steps):
        ep = probs(expert.feed(seq[expert.n:])[-1:])[0]
        if contrastive:
            ap = probs(amateur.feed(seq[amateur.n:])[-1:])[0]
            token = int(contrast(ep, ap)[2])
        else:
            token = int(ep.argmax())
        tokens.append(token)
        logp.append(float(np.log(ep[token])))
        seq.append(token)
    return tokens, logp


def contrast_data():
    """The worked example and the measurement over held-out passages.

    Worked example (:data:`CONTRAST_PROMPT`), over the whole vocabulary: ``top_*`` the
    expert's eight likeliest tokens (its probability, the amateur's, the difference, the
    contrast's probability after the filter), ``am_*`` the amateur's own five likeliest,
    ``raw_*`` the five largest differences when nothing is filtered, ``keep_*`` the tokens
    the filter keeps.

    Across :data:`PROMPTS` test passages. *One step*, at every position from the 8th on
    (arrays ``(prompt, position)``): ``differ`` whether the contrast's pick is not the
    expert's, ``hit_expert`` / ``hit_contrast`` whether each pick is the token the text
    really continues with, ``n_kept`` how many tokens pass the filter. *Written out*,
    :data:`ROLL` tokens after each whole passage (``(method, prompt)``, method 0 the
    expert's greedy choice, 1 the contrast's): ``loops``, ``distinct``, ``own`` as in
    :func:`penalty_data`; ``across_roll`` the tokens."""
    def build():
        expert, amateur = Stream(), Stream(AMATEUR_LAYERS)
        ids = _tok().encode(CONTRAST_PROMPT)
        out = expert.feed(ids, taps=(AMATEUR_LAYERS,))
        ep, ap = probs(out[gpt2.N_LAYER][-1:])[0], probs(out[AMATEUR_LAYERS][-1:])[0]
        plausible, score, _ = contrast(ep, ap)
        after = _softmax(np.where(plausible, score, -np.inf))
        top, am, raw = np.argsort(-ep)[:8], np.argsort(-ap)[:5], np.argsort(-score)[:5]
        keep = np.flatnonzero(plausible)
        keep = keep[np.argsort(-ep[keep])]
        worked = dict(
            top_tok=top, top_ep=ep[top], top_ap=ap[top], top_score=score[top], top_p=after[top],
            am_tok=am, am_ap=ap[am], am_ep=ep[am],
            raw_tok=raw, raw_ep=ep[raw], raw_ap=ap[raw], raw_score=score[raw],
            keep_tok=keep, keep_ep=ep[keep], keep_ap=ap[keep], keep_score=score[keep],
            keep_p=after[keep], threshold=np.asarray(ALPHA * ep.max()))
        prompts = heldout_prompts(PROMPTS, PROMPT_TOKENS)
        at = np.arange(8, PROMPT_TOKENS - 1)
        differ, hit_e, hit_c = (np.zeros((PROMPTS, len(at)), bool) for _ in range(3))
        n_kept = np.zeros((PROMPTS, len(at)), np.int64)
        loops = np.zeros((2, PROMPTS), bool)
        distinct, own = np.zeros((2, PROMPTS)), np.zeros((2, PROMPTS))
        across = np.zeros((2, PROMPTS, ROLL), np.int64)
        for pi, prompt in enumerate(prompts):
            expert.n = 0
            out = expert.feed(prompt, taps=(AMATEUR_LAYERS,))
            ep, ap = probs(out[gpt2.N_LAYER][at]), probs(out[AMATEUR_LAYERS][at])
            plausible, _, pick = contrast(ep, ap)
            differ[pi] = pick != ep.argmax(1)
            hit_e[pi], hit_c[pi] = ep.argmax(1) == prompt[at + 1], pick == prompt[at + 1]
            n_kept[pi] = plausible.sum(1)
            for m in range(2):
                tokens, logp = contrast_rollout(prompt, ROLL, expert, amateur, bool(m))
                across[m, pi] = tokens
                loops[m, pi] = repeats_trigram(prompt, tokens)
                distinct[m, pi] = distinct_fraction(tokens)
                own[m, pi] = np.mean(logp)
        return dict(worked, prompts=prompts, differ=differ, hit_expert=hit_e, hit_contrast=hit_c,
                    n_kept=n_kept, across_roll=across, loops=loops, distinct=distinct, own=own)
    return _cached("contrast", build, prompt=CONTRAST_PROMPT, layers=AMATEUR_LAYERS, alpha=ALPHA,
                   prompts=PROMPTS, length=PROMPT_TOKENS, roll=ROLL)


def contrast_summary(d=None):
    """The across-passages numbers of :func:`contrast_data` as plain counts and rates."""
    d = d or contrast_data()
    differ = d["differ"]
    return dict(
        positions=int(differ.size), differ=int(differ.sum()),
        hit_expert=int(d["hit_expert"].sum()), hit_contrast=int(d["hit_contrast"].sum()),
        # where the two picks differ: which of them was the real next token
        expert_right=int((differ & d["hit_expert"]).sum()),
        contrast_right=int((differ & d["hit_contrast"]).sum()),
        alone=int((d["n_kept"] == 1).sum()),
        loops=[int(v) for v in d["loops"].sum(1)],
        distinct=[float(v) for v in d["distinct"].mean(1)],
        own=[float(v) for v in d["own"].mean(1)])


def _contrastive():
    d = contrast_data()
    s = contrast_summary(d)
    labels = [show(t) for t in d["top_tok"]]
    n = s["positions"]
    return Study(
        title="Contrastive decoding",
        subtitle="GPT-2 small · four-layer amateur · one prompt, then %d held-out passages"
                 % PROMPTS,
        panels=(
            Panel("The expert", "The full twelve-layer model supplies the plausible token set.",
                  "bars", dict(values=d["top_ep"], labels=labels)),
            Panel("The amateur", "Stopping the same checkpoint at layer four gives a weaker "
                  "distribution.", "bars", dict(values=d["top_ap"], labels=labels)),
            Panel("Difference, then the plausibility filter",
                  "Expert minus amateur log-probability; tokens below 10% of the expert's "
                  "maximum are removed before the differences become probabilities.",
                  "bars", dict(values=d["top_score"], labels=labels, contrast=d["top_p"]),
                  ("Allowed tokens: %d" % len(d["keep_tok"]),
                   "Selected: " + show(d["keep_tok"][int(np.argmax(d["keep_p"]))]))),
            Panel("Across %d held-out passages" % PROMPTS,
                  "At %d positions: how often the contrast's pick differs from the expert's, "
                  "and how often each is the token the text really continues with. Then %d "
                  "tokens written each way: how many of the %d continuations loop."
                  % (n, ROLL, PROMPTS),
                  "bars",
                  dict(values=np.asarray([s["differ"] / n, s["hit_expert"] / n,
                                          s["hit_contrast"] / n, s["loops"][0] / PROMPTS,
                                          s["loops"][1] / PROMPTS]),
                       labels=["picks differ", "expert right", "contrast right",
                               "expert loops", "contrast loops"],
                       differ_each=d["differ"].mean(1), distinct=d["distinct"], own=d["own"]),
                  ("Where the picks differ (%d positions): expert right %d, contrast right %d"
                   % (s["differ"], s["expert_right"], s["contrast_right"]),)),
        ),
        metadata=_text_meta(
            source="Pinned GPT-2; prompts from the Tiny Shakespeare test split",
            settings=dict(amateur_layers=AMATEUR_LAYERS, alpha=ALPHA, prompts=PROMPTS,
                          prompt_tokens=PROMPT_TOKENS, new_tokens=ROLL),
            results=s,
            limitations=["Next-token accuracy is the expert's own training objective; the "
                         "written-out comparison is greedy, %d passages" % PROMPTS]),
    )


STUDIES = {
    "speculative-decoding": _speculative,
    "repetition-and-penalties": _penalties,
    "constrained-decoding": _constrained,
    "contrastive-decoding": _contrastive,
}

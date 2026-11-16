"""Seeded synthetic language and byte-pair tokenization for review studies."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

WORDS = ("red", "blue", "green", "small", "large", "cat", "dog", "bird", "runs", "sleeps")


def sentences(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return [
        "the %s %s %s." % (rng.choice(WORDS[:5]), rng.choice(WORDS[5:8]), rng.choice(WORDS[8:]))
        for _ in range(n)
    ]


def markov(n=128, length=17, seed=0, vocab=8, order=1):
    """Known stochastic language; the second mode changes the transition rule."""
    rng = np.random.default_rng(seed)
    x = rng.integers(vocab, size=(n, length))
    for t in range(1, length):
        follow = (x[:, t - 1] + order) % vocab
        x[:, t] = np.where(rng.random(n) < 0.85, follow, x[:, t])
    return x


def entropy(vocab=8, follow=0.85):
    p = np.full(vocab, (1 - follow) / vocab)
    p[0] += follow
    return float(-(p * np.log(p)).sum())


def bpe_train(text, merges=32):
    """Train deterministic BPE on characters; ties break lexicographically."""
    tokens = list(text)
    rules = []
    for _ in range(merges):
        counts = Counter(itertools.pairwise(tokens))
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        tokens = bpe_apply(tokens, [pair])
        rules.append(pair)
    return rules


def bpe_apply(tokens, rules):
    tokens = list(tokens)
    for a, b in rules:
        merged, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return tokens


def bigram_nll(train, test, alpha=0.25):
    """Held-out token NLL with add-alpha smoothing and unknown-token bucket."""
    vocab = sorted(set(train))
    ix = {s: i for i, s in enumerate(vocab)}
    unknown = len(ix)
    counts = np.full((unknown + 1, unknown + 1), alpha)
    for a, b in itertools.pairwise(train):
        counts[ix[a], ix[b]] += 1
    probs = counts / counts.sum(1, keepdims=True)
    return np.array(
        [
            -np.log(probs[ix.get(a, unknown), ix.get(b, unknown)])
            for a, b in itertools.pairwise(test)
        ]
    )

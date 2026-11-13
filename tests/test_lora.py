"""LoRA on GPT-2 small (:mod:`voxanim.lora`, :mod:`voxanim.lora_anim`): the
hand-written backward pass against finite differences (every weight of a
small random model, then the real adapters on the real loss), zero adapters
leaving the model untouched, the merge, the parameter count, and the claims
the animation makes about the recorded run (as relative claims: CI retrains
on another machine)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_gpt2

from voxanim import lora as LO

SMALL = LO.Config(n_layer=2, n_head=2, d_model=8, n_vocab=11, n_ctx=16)
SEQS = ([1, 4, 2, 7, 3, 9], [5, 5, 2], [10, 0, 3, 4, 8])      # three lengths: padding is exercised


def _small(seed=1, b_std=0.3):
    """A random small model, random adapters (``B`` non-zero, so ``A``'s
    gradient is not trivially zero) and a padded batch, in float64."""
    rng = np.random.default_rng(seed)
    W = LO.random_weights(SMALL, rng)
    L = LO.init_adapters(SMALL, rng, rank=3, dtype=np.float64, b_std=b_std)
    ids, w = LO.pad_batch(list(SEQS))
    return W, L, ids, w


# ------------------------------------------------------------------ the gradients (no model needed)


def test_every_gradient_matches_finite_differences_on_a_small_model():
    """Adapters and every base weight (embeddings, both LayerNorms of each
    layer, attention, MLP, the final LayerNorm, the tied unembedding) against
    central differences of the language-model loss."""
    W, L, ids, w = _small()
    alpha = 6.0

    def loss(Q):
        tr = LO.forward({k: Q[k] for k in W}, ids, {k: Q[k] for k in L}, SMALL, alpha)
        return LO.lm_loss(tr.logits, ids, w)[0]

    tr = LO.forward(W, ids, L, SMALL, alpha)
    value, dlogits = LO.lm_loss(tr.logits, ids, w)
    gL, gW = LO.backward(W, tr, dlogits, L, SMALL, alpha, frozen=True)
    P = dict(W, **L)
    G = dict(gW, **gL)
    assert set(G) == set(P)
    assert value == pytest.approx(loss(P))
    fd_check(loss, P, G, list(P), e=1e-6, tol=1e-6)


def test_padding_changes_nothing_for_the_real_tokens():
    """A sequence's logits are the same alone and right-padded in a batch
    (the causal mask keeps padding behind every real token)."""
    W, L, ids, w = _small()
    alone = LO.forward(W, np.array([SEQS[1]]), L, SMALL).logits[0]
    batched = LO.forward(W, ids, L, SMALL).logits[1, :len(SEQS[1])]
    assert np.allclose(alone, batched, atol=1e-12)
    # and padded targets carry no loss and no gradient
    _loss, dlogits = LO.lm_loss(LO.forward(W, ids, L, SMALL).logits, ids, w)
    assert np.all(dlogits[1, len(SEQS[1]) - 1:] == 0)


def test_zero_adapters_leave_the_model_unchanged_exactly():
    """With ``B = 0`` the adapted forward pass is bit for bit the base model's."""
    W, _L, ids, _w = _small()
    L0 = LO.init_adapters(SMALL, np.random.default_rng(3), rank=3, dtype=np.float64)
    assert all(np.all(v == 0) for k, v in L0.items() if k.endswith(".B"))
    assert np.array_equal(LO.forward(W, ids, L0, SMALL).logits,
                          LO.forward(W, ids, None, SMALL).logits)


def test_merged_weights_reproduce_the_adapted_model():
    """``W + (alpha / r) A B`` folded into ``c_attn`` gives the adapted logits,
    and only the adapted parts' columns change."""
    W, L, ids, _w = _small()
    M = LO.merge(W, L, SMALL, alpha=6.0)
    assert np.allclose(LO.forward(M, ids, None, SMALL).logits,
                       LO.forward(W, ids, L, SMALL, alpha=6.0).logits, atol=1e-12)
    d = SMALL.d_model
    for li in range(SMALL.n_layer):
        diff = M["h.%d.attn.c_attn.weight" % li] - W["h.%d.attn.c_attn.weight" % li]
        assert np.all(diff[:, d:2 * d] == 0)                     # K untouched
        assert np.abs(diff[:, :d]).max() > 0 and np.abs(diff[:, 2 * d:]).max() > 0
        assert np.linalg.matrix_rank(diff[:, :d]) == 3


def test_parameter_count():
    """12 layers x (Q, V) x (768 x 4 + 4 x 768) = 147,456 adapter numbers, 0.118%
    of GPT-2 small's 124,439,808 (counted from the arrays' shapes)."""
    L = LO.init_adapters(LO.GPT2, np.random.default_rng(0), rank=LO.RANK)
    assert LO.n_adapter_parameters(L) == 12 * 2 * (768 * 4 + 4 * 768) == 147_456
    small = LO.random_weights(SMALL, np.random.default_rng(0))
    assert LO.n_parameters(small) == sum(v.size for v in small.values())


# ------------------------------------------------------------------ the real model


@needs_gpt2
def test_gpt2_small_has_the_parameter_count_the_subtitle_quotes():
    from voxanim import gpt2_probe as P
    assert LO.n_parameters(P.weights()) == 124_439_808


@needs_gpt2
def test_real_adapter_gradients_match_finite_differences():
    """A few coordinates of the real adapters' gradients, on the real loss of
    two training sentences, against central differences in float64 (the
    weights cast once; ``B`` random so ``A``'s gradient is not zero)."""
    from voxanim import gpt2_probe as P
    W = {k: v.astype(np.float64) for k, v in P.weights().items()}
    rng = np.random.default_rng(0)
    L = LO.init_adapters(LO.GPT2, rng, dtype=np.float64, b_std=0.02)
    ids, w = LO.pad_batch([P.encode(s) for s in LO.SENTENCES[:2]])

    def loss(Q):
        return LO.lm_loss(LO.forward(W, ids, Q, LO.GPT2).logits, ids, w)[0]

    tr = LO.forward(W, ids, L, LO.GPT2)
    value, dlogits = LO.lm_loss(tr.logits, ids, w)
    G = LO.backward(W, tr, dlogits, L, LO.GPT2)
    assert 0 < value < 20
    keys = [LO.adapter_key(0, "q", "A"), LO.adapter_key(5, "q", "B"), LO.adapter_key(11, "v", "B"),
            LO.adapter_key(7, "v", "A")]
    fd_check(loss, L, G, keys, e=1e-4, tol=1e-4)


@needs_gpt2
def test_forward_matches_the_probe_and_zero_adapters_change_nothing_on_gpt2():
    """:func:`lora.forward` is the same computation as :func:`gpt2_probe.forward`,
    and fresh (``B = 0``) adapters leave GPT-2's logits exactly as they were."""
    from voxanim import gpt2_probe as P
    W = P.weights()
    ids = P.encode(LO.PROMPT)
    ours = LO.forward(W, np.array([ids]), None).logits[0]
    assert np.abs(ours - P.forward(ids)).max() < 2e-2            # float32 summation order differs
    L0 = LO.init_adapters(LO.GPT2, np.random.default_rng(0))
    assert np.array_equal(LO.forward(W, np.array([ids]), L0).logits[0], ours)


@needs_gpt2
def test_the_recorded_run_and_its_claims():
    """The claims the animation makes, as relative claims: the loss falls far,
    the target's probability rises by orders of magnitude to the top, the
    control prompts keep their top token, the update has rank 4, the merge
    reproduces the adapted logits to float32 precision."""
    from voxanim import gpt2_probe as P
    r = LO.run()
    assert r.steps == LO.STEPS and len(r.p_target) == LO.STEPS + 1
    assert r.loss[-1] < 0.35 * r.loss[0]                     # 4.43 -> 1.13 here
    assert r.p_target[0] < 1e-3 and r.p_target[-1] > 0.5
    assert r.p_target[-1] / r.p_target[0] > 1000
    assert r.top[0][0][0] != LO.TARGET and r.top[-1][0][0] == LO.TARGET
    # the control drawn (a two-shot fact), a story opening and a template with
    # another country keep their top token; the unrelated sentence the subtitle
    # quotes as "what else moved" does move (the old top token loses ground)
    for i in (0, 2, 4):
        _prompt, before, after = r.controls[i]
        assert before[0][0] == after[0][0], r.controls[i]
    _prompt, before, after = r.controls[1]
    p_after = dict(after).get(before[0][0], 0.0)
    assert p_after < 0.9 * before[0][1] or after[0][0] != before[0][0]
    assert np.all(r.B[0] == 0) and np.abs(r.B[-1]).max() > 0
    # rank 4: four singular values, the fifth zero to rounding
    assert r.sv[-1][3] > 1e-3 * r.sv[-1][0] and r.sv[-1][4] < 1e-5 * r.sv[-1][0]
    assert r.merge_max_abs < 1e-3 * r.merge_logit_scale
    assert r.n_adapter == 147_456 and r.n_base == 124_439_808
    # the merged model, built from the cached adapters, answers like the adapted one
    W = P.weights()
    ids = np.array([P.encode(LO.PROMPT)])
    merged = LO.forward(LO.merge(W, r.adapters), ids).logits[0, -1]
    adapted = LO.forward(W, ids, r.adapters).logits[0, -1]
    assert np.abs(merged - adapted).max() < 1e-3 * np.abs(adapted).max()
    p = np.exp(LO.log_softmax(merged))
    assert P.top_tokens(p, 1)[0][0] == LO.TARGET


@needs_gpt2
def test_animation_duration_phases_and_texts():
    from voxanim.lora_anim import LoRA
    a = LoRA()
    assert 45 <= a.duration <= 60
    texts = {tx.s for t in np.linspace(0.5, a.t_done, 40) for tx in a.scene(t).texts}
    assert any("147,456" in s for s in texts)
    assert any("rank 4" in s for s in texts)

"""Animations 1-6 of the "next 100": what they draw is what was measured."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_attention_a as M
from voxanim import next_attention as NA
from voxanim import presets, qa, realdata
from voxanim import study_attention_a as SA
from voxanim import tinylm as T
from voxanim.next_catalog import build_study

PRESETS = {
    "causal-vs-bidirectional": "causal-vs-bidirectional",
    "prefix-lm-attention": "prefix-lm-attention",
    "cross-attention": "cross-attention",
    "multi-head-vs-single": "multi-head-vs-single",
    "grouped-query-attention": "grouped-query-attention",
    "kv-cache": "gpt2-kv-cache",
}


def needs(name):
    return pytest.mark.skipif(not available(PRESETS[name]), reason="data not downloaded")


_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(PRESETS[name]).build(), build_study("next_attention", name))
    return _built[name]


def shown(anim, t):
    """Every string on screen at ``t``, with the live subtitle."""
    scene = anim.scene(float(t))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def phase_time(anim, k, u):
    a, b = anim.spans[k]
    return a + (b - a) * u


def has(strings, part):
    return any(part in s for s in strings)


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))
    key = lambda sc: sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)  # noqa: E731
    assert key(first) == key(last)
    assert len(first.lines) == len(last.lines)
    for k in range(len(anim.PHASES)):
        assert anim.subtitle_for(k)


# ---------------------------------------------------------------- 1 causal-vs-bidirectional
@needs("causal-vs-bidirectional")
def test_causal_masks_are_the_models_and_losses_the_studys():
    anim, study = built("causal-vs-bidirectional")
    configs = M.configs_of(study)
    assert np.array_equal(anim.causal, T.attention_mask(anim.N, configs[0]))
    assert np.array_equal(anim.full, T.attention_mask(anim.N, configs[1]))
    lo, hi = study.metadata["heldout_loss"]
    on = shown(anim, phase_time(anim, 3, 0.9))
    assert has(on, "%.2f nats" % lo) and has(on, "%.2f nats" % hi)
    assert "%d of %d pairs" % (anim.N * (anim.N + 1) // 2, anim.N ** 2) in anim.subtitle_for(0)
    assert anim.chars == [M.show(c) for c in realdata.decode(
        realdata.text_sequences(n=1, length=anim.N, split="test", seed=73)[0])]


# ---------------------------------------------------------------- 2 prefix-lm-attention
@needs("prefix-lm-attention")
def test_prefix_study_is_the_redesign_and_supports_its_claims():
    _anim, study = built("prefix-lm-attention")
    m = study.metadata
    assert study.title == SA.prefix_lm_attention().title and "designs_tried" in m
    accuracy, loss = np.array(m["written_accuracy"]), np.array(m["heldout_loss"])
    assert accuracy.shape == loss.shape == (3, len(SA.SEEDS)) and len(SA.SEEDS) >= 3
    causal, prefix, both = accuracy
    # a reply that is causal can be written; one trained reading ahead cannot
    assert causal.min() > 0.9 and prefix.min() > 0.9
    assert both.max() < min(causal.min(), prefix.min()) - 0.5
    # ... although shown the true reply it scores best: it reads its own answer
    assert loss[2].max() < min(loss[0].min(), loss[1].min()) - 0.02
    # prefix against causal: no claim either way beyond a point and a half
    assert abs(prefix.mean() - causal.mean()) < 0.015
    assert np.allclose(study.panels[2].data["values"], accuracy.mean(1))
    assert np.allclose(study.panels[1].data["ys"],
                       np.array(m["written_accuracy_curves"]).mean(1))
    assert np.allclose(np.array(m["written_accuracy_curves"])[:, :, -1], accuracy)


@needs("prefix-lm-attention")
def test_prefix_data_is_real_text_and_held_out():
    prompts, x, y, w = SA.prefix_heldout()
    k = SA.PROMPT
    test, train = realdata.text("test"), realdata.text("train")
    assert test not in train and len(prompts) == 512
    for row in prompts[:32]:
        assert realdata.decode(row) in test
    assert np.array_equal(y[:, k:], np.sort(prompts, 1))            # the reply: the prompt sorted
    assert np.array_equal(x[:, :k], prompts) and (x[:, k] == 0).all()
    assert np.array_equal(x[:, k + 1:], y[:, k:-1])                  # the reply so far, shifted
    assert w[0].tolist() == [0.0] * k + [1.0] * k
    runs = SA.prefix_runs()
    for name in SA.MASKS:
        assert [r["settings"]["seed"] for r in runs[name]] == list(SA.SEEDS)
        run = runs[name][0]
        # written alone: recomputed here from the trained parameters
        again = SA.write_reply(run["P"], run["config"], prompts[:16])
        assert np.array_equal(again, run["written"][:16])


@needs("prefix-lm-attention")
def test_prefix_masks_are_the_models():
    anim, study = built("prefix-lm-attention")
    assert np.array_equal(anim.masks[1], study.panels[0].data["values"])
    n, k = anim.N, anim.K
    assert (n, k) == (16, 8)
    i, j = np.indices((n, n))
    assert np.array_equal(anim.masks[0], j <= i)
    assert np.array_equal(anim.masks[1], (j <= i) | ((i < k) & (j < k)))
    assert np.array_equal(anim.masks[2], np.ones((n, n)))
    assert anim.pairs == [136, 164, 256]
    assert has(shown(anim, phase_time(anim, 1, 0.95)), "164 of 256 pairs")


@needs("prefix-lm-attention")
def test_prefix_numbers_shown_are_the_studys():
    anim, study = built("prefix-lm-attention")
    accuracy = np.array(study.metadata["written_accuracy"])
    on = shown(anim, phase_time(anim, 4, 0.95))
    for row in accuracy:
        assert has(on, "%.1f%%  (seeds %.1f to %.1f)" % (
            100 * row.mean(), 100 * row.min(), 100 * row.max()))
    overlap = not (accuracy[1].min() > accuracy[0].max() or accuracy[0].min() > accuracy[1].max())
    assert has(on, "the seeds overlap") == overlap == (not anim.clear)
    assert has(on, "loss %.3f nats" % np.mean(study.metadata["heldout_loss"][2]))
    # the prompt drawn is the first held-out one, and each row what that mask's model wrote
    prompts, _x, y, _w = SA.prefix_heldout()
    letters = lambda ids: [M.show(c) for c in realdata.decode(ids)]  # noqa: E731
    assert anim.prompt == letters(prompts[0]) and anim.target == letters(y[0, anim.K:])
    runs = SA.prefix_runs()
    writing = shown(anim, phase_time(anim, 2, 0.95))
    for name, wrote in zip(SA.MASKS, anim.written):
        assert wrote == letters(runs[name][0]["written"][0])
        assert has(writing, name + " writes")
    right = [sum(a == b for a, b in zip(w, anim.target)) for w in anim.written]
    for n in right:
        assert has(writing, "%d of 8 right" % n)


# ---------------------------------------------------------------- 3 cross-attention
@needs("cross-attention")
def test_cross_attention_drawn_is_the_trained_models():
    anim, study = built("cross-attention")
    # the model rebuilt here is the study's: its first span's head is the study's panel
    assert np.allclose(anim.model_attention[0, 0], study.panels[1].data["values"], atol=1e-12)
    assert np.array_equal(anim.attn, anim.model_attention[anim.EXAMPLE, 0])
    assert np.allclose(anim.attn.sum(1), 1.0)
    source, _x, target = NA.denoising_data(64, "test", 73)
    assert anim.blank.tolist() == (source[anim.EXAMPLE] == 0).tolist()
    assert anim.blank.tolist() == [False, True, False, False, True, False]
    assert anim.target == [M.show(c) for c in realdata.decode(target[anim.EXAMPLE])]
    # the stages are that run on its way: none at the start, the study's at the end
    assert [n for n, _ in anim.stages] == [0, 20, 60, 120, 200, 300]
    assert np.array_equal(anim.stages[-1][1], anim.attn)
    assert not np.allclose(anim.stages[0][1], anim.attn, atol=0.05)


@needs("cross-attention")
def test_cross_attention_numbers_shown():
    anim, study = built("cross-attention")
    acc = study.metadata["accuracy"]
    on = shown(anim, phase_time(anim, 3, 0.95))
    assert has(on, "%.1f%%" % (100 * acc))
    assert has(on, "%.1f%%" % (100 * study.panels[3].data["values"][1]))
    # seen and blanked positions together are every position
    assert np.isclose((4 * anim.seen_accuracy + 2 * anim.blank_accuracy) / 6, acc)
    assert has(on, "%.1f%%" % (100 * anim.seen_accuracy))
    assert has(on, "%.1f%%" % (100 * anim.blank_accuracy))
    loss = np.asarray(study.panels[2].data["ys"])[0]
    assert "%.2f to %.2f" % (loss[0], loss[-1]) in anim.subtitle_for(2)
    mid = shown(anim, phase_time(anim, 1, 0.45))
    assert has(mid, "%.2f" % anim.attn[2].max())


# ---------------------------------------------------------------- 4 multi-head-vs-single
@needs("multi-head-vs-single")
def test_heads_study_is_the_redesign_and_supports_its_claims():
    _anim, study = built("multi-head-vs-single")
    m = study.metadata
    assert "designs_tried" in m and m["parameters"] == [42624, 42624]
    loss = np.array(m["heldout_loss"])
    assert loss.shape == (2, len(SA.SEEDS)) and len(SA.SEEDS) >= 3
    one, eight = loss
    # neither model memorised: far below guessing evenly, on text it never trained on
    assert loss.max() < m["guessing_evenly"] - 1.5
    # eight heads are better on average, by more than a hundredth of a nat
    assert eight.mean() < one.mean() - 0.01
    assert np.allclose(study.panels[3].data["values"], loss.mean(1))
    assert np.allclose(study.panels[2].data["values"], loss.ravel())
    # the heads differ from each other in where they look, in every seed
    for own, before in zip(m["self_share"][1], m["previous_character_share"][1]):
        assert len(own) == 8
        assert max(before) - min(before) > 0.15 and max(own) - min(own) > 0.10
    # ... against a one-head model's single mixture, which hardly moves between seeds
    single = [v[0] for v in m["previous_character_share"][0]]
    assert max(single) - min(single) < 0.05
    table = np.asarray(study.panels[0].data["values"])
    assert table.shape == (9, 6) and np.allclose(table.sum(1), 1.0)


@needs("multi-head-vs-single")
def test_heads_data_is_real_text_and_held_out():
    x, y = SA.head_heldout()
    test, train = realdata.text("test"), realdata.text("train")
    assert test not in train and x.shape == (512, SA.CONTEXT)
    assert np.array_equal(x[:, 1:], y[:, :-1])                      # targets: the next character
    for row in x[:32]:
        assert realdata.decode(row) in test
    runs = SA.head_runs()
    for heads in SA.HEAD_COUNTS:
        assert [r["settings"]["seed"] for r in runs[heads]] == list(SA.SEEDS)
        run = runs[heads][0]
        assert np.isclose(T.loss(run["P"], x, y, run["config"]), run["heldout_loss"])
        assert run["attention"].shape == (512, heads, SA.CONTEXT, SA.CONTEXT)
    # an even spread over the i+1 readable positions, readers 5..15
    even = SA.even_spread()
    assert np.isclose(even[0], np.mean([1 / (i + 1) for i in range(SA.FROM, SA.CONTEXT)]))
    assert np.isclose(even.sum(), 1.0)


@needs("multi-head-vs-single")
def test_heads_drawn_are_the_trained_models():
    anim, study = built("multi-head-vs-single")
    table = np.asarray(study.panels[0].data["values"])
    assert np.allclose(anim.offsets, table, atol=1e-12)
    assert anim.one.shape == (16, 16) and anim.eight.shape == (8, 16, 16)
    assert np.allclose(anim.one.sum(1), 1.0) and np.allclose(anim.eight.sum(2), 1.0)
    assert np.allclose(np.triu(anim.one, 1), 0.0)           # causal: nothing read ahead
    x, _y = SA.head_heldout()
    assert anim.chars == [M.show(c) for c in realdata.decode(x[0])]
    # the table's first column is what the drawn attention maps average to
    runs = SA.head_runs()
    readers = np.arange(SA.FROM, SA.CONTEXT)
    assert np.allclose(runs[8][0]["attention"][:, :, readers, readers].mean((0, 2)), table[1:, 0])
    on = shown(anim, phase_time(anim, 2, 0.95))
    for v in table[:, 1]:
        assert has(on, "%.0f%%" % (100 * v))
    # the other seeds' ranges are said too, not only the drawn model's
    for lo, hi in anim.before[1:]:
        assert "%.0f%% to %.0f%%" % (lo, hi) in anim.subtitle_for(2)


@needs("multi-head-vs-single")
def test_head_numbers_shown():
    anim, study = built("multi-head-vs-single")
    loss = np.array(study.metadata["heldout_loss"])
    on = shown(anim, phase_time(anim, 4, 0.95))
    assert has(on, "mean %.3f" % loss[0].mean()) and has(on, "mean %.3f" % loss[1].mean())
    assert has(on, "%.3f nats lower on average" % (loss[0].mean() - loss[1].mean()))
    assert has(on, "guessing evenly costs %.2f nats" % np.log(len(realdata.vocabulary())))
    assert anim.clear == bool(loss[1].max() < loss[0].min())
    assert has(on, "is below every one-head model") == anim.clear
    assert "42,624 parameters" in anim.subtitle_for(3)
    assert has(shown(anim, phase_time(anim, 3, 0.9)), "thin lines: each of 3 seeds")


# ---------------------------------------------------------------- 5 grouped-query-attention
@needs("grouped-query-attention")
def test_gqa_memory_is_the_exact_count():
    anim, study = built("grouped-query-attention")
    cls = M.GroupedQueryAttention
    exact = [cls.LAYERS * kv * 2 * cls.HEAD_WIDTH * cls.BYTES for kv in cls.KV_HEADS]
    assert exact == [int(v) for v in study.panels[0].data["values"]] == anim.per_token
    on = shown(anim, phase_time(anim, 1, 0.95))
    for v in exact:
        assert has(on, f"{v:,} bytes")
    end = shown(anim, phase_time(anim, 2, 0.95))
    assert has(end, "context: 8,192 tokens")
    for v in exact:
        assert has(end, M.si_bytes(v * cls.CONTEXT))
    assert M.si_bytes(131072 * 8192) == "1.07 GB"
    # the sharing drawn is the model's: head i reads key/value head i * kv // heads
    for kv, groups in ((8, 8), (2, 2), (1, 1)):
        assert len({i * kv // cls.QUERY_HEADS for i in range(cls.QUERY_HEADS)}) == groups


@needs("grouped-query-attention")
def test_gqa_numbers_shown():
    anim, study = built("grouped-query-attention")
    on = shown(anim, phase_time(anim, 4, 0.95))
    for v in study.metadata["heldout_loss"]:
        assert has(on, "%.3f nats" % v)
    for n in study.panels[3].data["values"]:
        assert has(on, f"{int(n):,}")
    assert (anim.small_q, anim.small_kv) == (4, [4, 2, 1])


# ---------------------------------------------------------------- 6 kv-cache
@needs("kv-cache")
def test_kv_cache_demo_is_gpt2():
    from voxanim import gpt2, gpt2_hooks

    anim, _study = built("kv-cache")
    d = anim.demo
    tok = gpt2.Tokenizer()
    held_out = tok.encode(realdata.text("test")[:64])
    ids = [int(i) for i in d["ids"]]
    assert ids == [int(i) for i in held_out[6:6 + len(ids)]]
    full = gpt2_hooks.forward(ids)
    assert np.array_equal(full["logits"].argmax(1), d["full_next"])
    assert np.array_equal(d["full_next"], d["cached_next"])
    assert anim.agree == anim.T == 8
    assert float(d["diff"].max()) < 1e-3 < float(d["peak"])
    key = np.asarray(full["cache"][0][0]).transpose(1, 0, 2).reshape(len(ids), -1)
    assert np.allclose(anim.key, key[:, :M.KV_SHOWN])
    # a key computed once and kept is the key computed again: the cache's point
    assert np.allclose(d["cached_key"], d["full_key"], atol=1e-4)
    assert np.allclose(d["cached_value"], d["full_value"], atol=1e-4)
    assert key.shape[1] == anim.WIDTH


@needs("kv-cache")
def test_kv_cache_numbers_shown():
    anim, study = built("kv-cache")
    n = len(study.panels[1].data["x"])
    assert anim.total == [n * (n + 1) // 2, n] == [351, 26]
    end = shown(anim, phase_time(anim, 3, 0.95))
    assert has(end, "351 computed") and has(end, "26 computed")
    assert anim.per_token == 2 * anim.LAYERS * anim.WIDTH * 4
    memory = shown(anim, phase_time(anim, 4, 0.95))
    assert has(memory, "26 tokens: {:,} bytes".format(int(study.panels[2].data["ys"][0][-1])))
    assert "%.6f" % np.max(study.panels[0].data["ys"]) in anim.subtitle_for(2)
    first = shown(anim, phase_time(anim, 0, 0.9))
    assert has(first, "step 8: 8 rows computed, 36 so far")
    second = shown(anim, phase_time(anim, 1, 0.9))
    assert has(second, "step 8: 1 row computed, 8 so far")

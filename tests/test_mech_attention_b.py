"""Animations 7-12 of the "next 100": what they draw is what was measured."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_attention_b as M
from voxanim import presets, qa, realdata
from voxanim import study_attention_b as S
from voxanim import tinylm as T
from voxanim.next_attention import sparse_mask
from voxanim.next_catalog import build_study

PRESETS = {
    "sliding-window-attention": "sliding-window-attention",
    "sparse-attention-patterns": "sparse-attention-patterns",
    "linear-attention": "linear-attention",
    "flash-attention-tiling": "flash-attention-tiling",
    "attention-sinks": "gpt2-attention-sinks",
    "attention-temperature": "attention-temperature",
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


# ---------------------------------------------------------------------- 7
@needs("sliding-window-attention")
def test_sliding_window_reach_is_the_composed_mask():
    anim, study = built("sliding-window-attention")
    band, reach = M.reach_after(12, 4, 3)
    assert np.array_equal(band, T.attention_mask(12, T.Config(mask="window", window=4)))
    assert np.array_equal(reach[-1], np.asarray(study.panels[0].data["values"]) > 0)
    assert [int(r[-1].sum()) for r in reach] == [4, 7, 10] == anim.reached
    # the cone drawn for depth L lights exactly the characters the mask says are reached
    for layers, r in enumerate(reach, 1):
        need = [max(1, int(np.ceil((11 - j) / 3))) for j in range(12)]
        assert [n <= layers for n in need] == list(r[-1])
    # "reach covers the distance" in the study is that same composition, on the trained mask
    m = study.metadata
    trained = T.attention_mask(m["length"], S.window_config(1))
    last = m["length"] - 1
    for a, layers in enumerate(m["layers"]):
        for b, delay in enumerate(m["delays"]):
            assert S.composed(trained, layers)[last, last - delay] == m["reach_covers"][a][b]
    assert m["reach_back"] == [3, 6, 9] == anim.back


def test_the_echo_task_asks_for_real_characters_from_disjoint_text():
    if not available("sliding-window-attention"):
        pytest.skip("data not downloaded")
    x, y, w = S.echo_task(64, 20, 6, "train", S.TRAIN_SEED)
    assert np.array_equal(y[:, 6:], x[:, :-6])           # the target is the character 6 back
    assert not w[:, :6].any() and w[:, 6:].all()         # and only asked where there is one
    assert not y[:, :6].any()
    train, test = S._provenance("x")["dataset"], S._provenance("x")["evaluation_dataset"]
    assert train["split"] == "train" and test["split"] == "test"
    assert train["character_range"][1] <= test["character_range"][0]      # disjoint text
    xt, _, _ = S.echo_task(S.TEST_WINDOWS, 20, 6, "test", S.TEST_SEED)
    assert realdata.decode(xt[0]) in realdata.text("test")               # unmodified text


@needs("sliding-window-attention")
def test_sliding_window_models_copy_from_exactly_as_far_as_they_reach():
    """The claim the redesigned study supports: on held-out text a stack answers
    where its composed mask covers the distance, on every seed, and not one
    place further."""
    _anim, study = built("sliding-window-attention")
    m = study.metadata
    acc = np.asarray(m["heldout_accuracy"])              # depth, distance, seed
    covers = np.asarray(m["reach_covers"])
    assert acc.shape == (3, 6, 3) and len(m["seeds"]) == 3
    assert m["delays"] == [3, 4, 6, 7, 9, 10]
    assert acc[covers].min() > 0.8                       # every covered model, every seed
    assert acc[~covers].max() < 0.4
    assert acc[covers].min() > acc[~covers].max() + 0.4
    for depth, (limit, beyond) in enumerate(((0, 1), (2, 3), (4, 5))):
        # the control: the same depth, asked one place past its reach
        assert acc[depth, limit].min() > acc[depth, beyond].max() + 0.5
    # depth is what buys the distance: at 9 back only three layers answer
    nine = m["delays"].index(9)
    assert acc[2, nine].min() > max(acc[0, nine].max(), acc[1, nine].max()) + 0.5
    loss = np.asarray(m["heldout_loss"])
    assert loss[covers].max() < loss[~covers].min() - 1.0
    assert len(m["checkpoints"]) == 54 and len(m["designs_tried"]) >= 2
    assert np.allclose(study.panels[2].data["ys"], acc.mean(2))
    assert np.allclose(study.panels[3].data["values"], loss.mean(2))


@needs("sliding-window-attention")
def test_sliding_window_numbers_are_the_studys():
    anim, study = built("sliding-window-attention")
    m = study.metadata
    acc = np.asarray(m["heldout_accuracy"])
    covers = np.asarray(m["reach_covers"])
    assert "42 of 144 pairs" in anim.subtitle_for(0)
    assert "%.0f to %.0f%% right where the stack's reach covers" % (
        100 * acc[covers].min(), 100 * acc[covers].max()) in anim.subtitle_for(4)
    assert "%.0f to %.0f%% where it does not" % (
        100 * acc[~covers].min(), 100 * acc[~covers].max()) in anim.subtitle_for(4)
    texts = shown(anim, phase_time(anim, 4, 0.95))
    for value in acc.mean(2).ravel():
        assert "%.0f%%" % (100 * value) in texts
    for delay in m["delays"]:
        assert "%d back" % delay in texts
    assert np.array_equal(anim.traces, study.panels[1].data["ys"])
    assert np.allclose(anim.traces, np.asarray(study.panels[1].data["each"]).mean(1))
    assert "%.2f, %.2f and %.2f nats" % tuple(anim.traces[:, -20:].mean(1)) in anim.subtitle_for(3)


@needs("sliding-window-attention")
def test_sliding_window_task_rows_are_the_models_answers():
    anim, study = built("sliding-window-attention")
    m = study.metadata
    window = m["example"]["window"]
    first = S.echo_task(S.TEST_WINDOWS, m["length"], 6, "test", S.TEST_SEED)[0][0]
    assert window == list(first)                         # the first held-out window, as it is
    assert anim.window == M.glyphs(window) and anim.chars == anim.window[-12:]
    for turn, delay in enumerate(anim.ASKED):
        texts = shown(anim, phase_time(anim, 2, (turn + 0.9) / 3 / 1.03))
        assert "name the character %d back" % delay in texts
        col = m["delays"].index(delay)
        for depth in range(3):
            said = m["example"]["said"][depth][col]
            right = sum(said[i] == window[i - delay] for i in range(delay, len(window)))
            assert "%d of %d right" % (right, len(window) - delay) in texts


# ---------------------------------------------------------------------- 8
@needs("sparse-attention-patterns")
def test_sparse_masks_are_the_trained_masks():
    anim, study = built("sparse-attention-patterns")
    m = study.metadata
    n = m["length"]
    assert n == 32 and m["kinds"] == list(anim.KINDS)
    stacked = np.concatenate([sparse_mask(n, kind) for kind in anim.KINDS]).astype(float)
    assert np.array_equal(np.concatenate(anim.full).astype(float), stacked)
    assert np.array_equal(stacked, study.panels[0].data["values"])
    assert list(study.panels[2].data["labels"]) == list(anim.KINDS)
    assert anim.pairs == [78, 54, 30, 50, 56]              # the rules as drawn, at 12 characters
    assert anim.full_pairs == m["pairs"] == [int(v.sum()) for v in anim.full]
    assert m["pairs"][0] == 32 * 33 // 2 and m["pairs"][0] > 2 * m["pairs"][1]
    for mask in anim.masks + anim.full:           # every pattern is causal and reads itself
        assert not np.triu(mask, 1).any() and mask.diagonal().all()


@needs("sparse-attention-patterns")
def test_sparse_reach_is_two_hops_of_the_mask():
    anim, study = built("sparse-attention-patterns")
    m = study.metadata
    asked, delay = m["asked_positions"], m["delay"]
    assert asked == list(range(delay, m["length"])) and m["layers"] == 2
    for k, mask in enumerate(anim.full):
        two = (mask.astype(int) @ mask.astype(int)) > 0
        assert [int(two[i, i - delay]) for i in asked] == m["reachable"][k]
        for r, q in enumerate(asked):              # the route drawn exists exactly when it can
            hops = anim.route(k, q)
            assert bool(hops) == bool(m["reachable"][k][r])
            if len(hops) == 1:
                assert mask[q, q - delay]
            if len(hops) == 2:
                assert mask[q, hops[0]] and mask[hops[0], q - delay] and hops[1] == q - delay
    share = dict(zip(anim.KINDS, m["reachable_share"]))
    assert share["dense"] == share["strided"] == 1.0 and share["block"] == 0.0
    assert 0.0 < share["global"] < 0.1 < share["random"] < 1.0


@needs("sparse-attention-patterns")
def test_sparse_patterns_score_where_their_two_hops_reach():
    """The claim the redesigned study supports: what a pattern saves in pairs it
    pays for in what it can connect - and strided connects everything dense
    does with under half the pairs."""
    anim, study = built("sparse-attention-patterns")
    m = study.metadata
    acc = dict(zip(anim.KINDS, np.asarray(m["heldout_accuracy"])))       # per seed
    assert all(len(v) == 3 for v in acc.values())
    assert acc["dense"].min() > 0.95
    assert max(acc["block"].max(), acc["global"].max()) < 0.4
    assert acc["dense"].min() > max(acc["block"].max(), acc["global"].max()) + 0.5
    assert acc["strided"].min() > max(acc["block"].max(), acc["global"].max()) + 0.3
    assert acc["random"].min() > max(acc["block"].max(), acc["global"].max()) + 0.15
    assert acc["random"].max() < acc["dense"].min() - 0.15
    # the exact control: the same models, at the askers their mask can and cannot connect
    within = np.asarray(m["right_where_reachable"]).sum() / (3 * sum(m["positions_reachable"]))
    beyond = np.asarray(m["right_where_unreachable"]).sum() / (
        3 * sum(m["positions_unreachable"]))
    assert within > beyond + 0.4
    assert np.isclose(within, anim.right_within) and np.isclose(beyond, anim.right_beyond)
    k = anim.KINDS.index("random")                         # within one pattern, too
    assert (np.asarray(m["right_where_reachable"])[k] / m["positions_reachable"][k]).min() > (
        np.asarray(m["right_where_unreachable"])[k] / m["positions_unreachable"][k]).max() + 0.2
    for kind, pattern in zip(anim.KINDS, range(5)):        # totals are the accuracy reported
        right = np.asarray(m["right_where_reachable"])[pattern] + np.asarray(
            m["right_where_unreachable"])[pattern]
        assert np.allclose(right / (len(m["asked_positions"]) * S.TEST_WINDOWS), acc[kind])
    assert len(m["checkpoints"]) == 15 and len(m["designs_tried"]) >= 2


@needs("sparse-attention-patterns")
def test_sparse_numbers_are_the_studys():
    anim, study = built("sparse-attention-patterns")
    m = study.metadata
    mean = np.asarray(study.panels[2].data["values"], float)
    assert np.allclose(mean, np.asarray(m["heldout_accuracy"]).mean(1))
    for kind, value, pairs in zip(anim.KINDS, mean, m["pairs"]):
        assert "%s %.0f%% right with %d pairs" % (kind, 100 * value, pairs) in anim.subtitle_for(4)
    texts = shown(anim, phase_time(anim, 4, 0.95))
    for value, pairs in zip(mean, m["pairs"]):
        assert "%.0f%% · %d pairs" % (100 * value, pairs) in texts
    counts = np.asarray(study.panels[3].data["ys"])
    assert int(counts[0, -1]) == 65536 * 65537 // 2             # dense: every causal pair
    texts = shown(anim, phase_time(anim, 1, 0.95))
    assert "at 65,536 characters (exact counts)" in texts
    assert "%s pairs" % M.number(counts[0, -1]) in texts
    assert np.array_equal(anim.traces, study.panels[1].data["ys"])
    # "two hops": each pattern's turn states its exact reach and its model's answers
    window, delay = m["example"]["window"], m["delay"]
    for k, kind in enumerate(anim.KINDS):  # noqa: B007
        texts = " | ".join(shown(anim, phase_time(anim, 2, (k + 0.6) / 5 / 1.02)))
        assert "(exact): %d of 22" % sum(m["reachable"][k]) in texts
        said = m["example"]["said"][k]
        right = sum(said[i] == window[i - delay] for i in m["asked_positions"])
        assert "(filled: right): %d of 22" % right in texts


# ---------------------------------------------------------------------- 9
def test_kernel_running_sum_is_kernel_attention():
    x = M.kernel_example()
    assert np.abs(x["pair"] - x["rec"]).max() < 1e-13           # the regrouping is exact
    assert np.allclose(x["kernel"].sum(1), 1) and np.allclose(x["softmax"].sum(1), 1)
    assert (x["fk"] > 0).all() and (x["fq"] > 0).all()
    # the sum after each key is the sum of the keys so far
    assert np.allclose(x["S"][4], sum(np.outer(x["fk"][j], x["v"][j]) for j in range(5)))
    assert np.allclose(x["z"][-1], x["fk"].sum(0))


@needs("linear-attention")
def test_linear_attention_numbers_are_the_studys():
    anim, study = built("linear-attention")
    identity, gap = [float(v) for v in study.panels[0].data["values"]]
    x = anim.x
    assert np.abs(x["pair"] - x["soft_out"]).max() == pytest.approx(gap, abs=1e-12)
    assert identity < 1e-13
    assert "%.1e" % identity in anim.subtitle_for(2) and "%.2f" % gap in anim.subtitle_for(2)
    lo, hi = study.metadata["heldout_loss"]
    assert "%.2f nats for softmax, %.2f for the kernel" % (lo, hi) in anim.subtitle_for(3)
    texts = shown(anim, phase_time(anim, 3, 0.97))
    assert "%.2f" % lo in texts and "%.2f" % hi in texts
    assert "32 steps, not 256 scores" in anim.subtitle_for(1)


# ---------------------------------------------------------------------- 10
def test_flash_rows_are_the_online_softmax():
    x = M.flash_example()
    rows = x["rows"]
    assert np.allclose(rows[:, [1, 4]], x["hist"])              # running max and sum, per tile
    scores = x["scores"][0]
    assert rows[-1, 1] == scores.max()
    assert rows[-1, 4] == pytest.approx(np.exp(scores - scores.max()).sum(), rel=1e-12)
    assert rows[0, 2] == 0.0                                     # nothing to rescale at first
    assert (rows[1:, 2] <= 1.0).all()
    assert np.abs(x["dense"] - x["tiled"]).max() < 1e-13


def test_flash_numbers_are_the_studys():
    anim, study = built("flash-attention-tiling")
    assert np.allclose(anim.x["rows"][:, [1, 4]], np.asarray(study.panels[1].data["ys"]).T)
    assert np.allclose(anim.x["weights"], study.panels[0].data["values"])
    error = float(study.panels[2].data["values"][0])
    assert np.abs(anim.x["dense"] - anim.x["tiled"]).max() == pytest.approx(error, abs=1e-18)
    assert "%.1e" % error in anim.subtitle_for(2)
    assert "%.1e" % error in shown(anim, phase_time(anim, 2, 0.95))
    texts = shown(anim, phase_time(anim, 1, 0.95))
    for m, total in np.asarray(study.panels[1].data["ys"]).T:
        assert "%.2f" % m in texts and "%.2f" % total in texts
    sizes = np.asarray(study.panels[3].data["ys"])
    assert "33.6M bytes for every score, 131,072 for one tile" in anim.subtitle_for(3)
    assert sizes[0, -1] == 8 * 2048 ** 2 and sizes[1, -1] == 8 * 2048 * 8


# ---------------------------------------------------------------------- 11
def test_cache_slots_follow_the_studys_eviction():
    for policy in (0, 1, 2):
        kept = []
        for i in range(63):                       # the study's loop, positions only
            if policy and len(kept) >= 32:
                keep = list(range(len(kept) - 31, len(kept)))
                if policy == 2:
                    keep = list(range(4)) + list(range(len(kept) - 27, len(kept)))
                kept = [kept[j] for j in keep]
            kept.append(i)
            assert M.cache_slots(policy, i) == kept
    assert len(M.cache_slots(1, 62)) == 32 == len(M.cache_slots(2, 62))
    assert 0 not in M.cache_slots(1, 40) and M.cache_slots(2, 40)[:4] == [0, 1, 2, 3]


@needs("attention-sinks")
def test_sink_attention_is_the_studys():
    anim, study = built("attention-sinks")
    mass = np.asarray(study.panels[0].data["values"], float)
    late = anim.attn[:, :, anim.window:, 0].mean(2)             # the study's late queries
    assert np.abs(late - mass).max() < 1e-5
    assert np.allclose(anim.attn[:, :, anim.reader, :anim.reader + 1].sum(-1), 1, atol=1e-4)
    assert anim.heads[2] == np.unravel_index(mass.argmax(), mass.shape)
    assert anim.over_half == int((mass > 0.5).sum())
    assert "%d of GPT-2's 144 heads" % anim.over_half in anim.subtitle_for(1)


@needs("attention-sinks")
def test_sink_numbers_are_the_studys():
    anim, study = built("attention-sinks")
    full, last, sinks = study.metadata["perplexity"]
    assert "%.0f with everything cached" % full in anim.subtitle_for(3)
    assert "%.0f with 4 of them kept" % sinks in anim.subtitle_for(3)
    texts = shown(anim, phase_time(anim, 3, 0.95))
    for value in (full, last, sinks):
        assert M.commas(value) in texts
    assert ["%d slots" % v for v in study.panels[3].data["values"]] == [
        t for t in texts if t.endswith("slots")]
    # the running perplexity read out beside the strips ends at the suffix perplexity
    assert np.allclose(anim.running[:, -1], [full, last, sinks])
    texts = shown(anim, phase_time(anim, 2, 0.95))
    assert "last 32  %s" % M.commas(last) in texts


# ---------------------------------------------------------------------- 12
def test_scale_example_spreads_like_the_root_of_the_width():
    x = M.scale_example()
    assert np.allclose(x["spread"] / np.sqrt(x["d"]), 1.0, atol=0.1)
    assert np.allclose(x["raw"].sum(1), 1) and np.allclose(x["scaled"].sum(1), 1)
    assert (x["hs"] > 2.9).all() and (np.diff(x["h"]) < 0).all()
    assert (x["gs"] > x["g"]).all()


@needs("attention-temperature")
def test_score_scale_numbers_are_the_studys():
    anim, study = built("attention-temperature")
    h, hs = np.asarray(study.panels[0].data["ys"], float)
    assert np.allclose(anim.x["h"], h) and np.allclose(anim.x["hs"], hs)
    assert np.allclose(anim.x["g"], np.asarray(study.panels[3].data["ys"])[0])
    assert np.array_equal(anim.x["d"], study.panels[0].data["x"])
    assert "falls from %.2f to %.2f" % (h[0], h[-1]) in anim.subtitle_for(2)
    scaled, unscaled = study.metadata["heldout_loss"]
    assert "%.2f nats with the division, %.2f without" % (scaled, unscaled) in anim.subtitle_for(4)
    assert list(study.panels[2].data["labels"]) == list(anim.NAMES)
    texts = shown(anim, phase_time(anim, 4, 0.97))
    assert "%.2f" % scaled in texts and "%.2f" % unscaled in texts
    assert anim.head == 4

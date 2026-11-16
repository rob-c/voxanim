"""The drawn animations 13-20 (:mod:`voxanim.mech_block`): layout, the numbers
they show against their studies, and the data they compute themselves."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_block as M
from voxanim import next_attention as NA
from voxanim import presets, qa
from voxanim import realdata as R
from voxanim import study_block as SB
from voxanim import tinylm as T
from voxanim.next_catalog import build_study

PRESETS = {
    "pre-ln-vs-post-ln": "pre-ln-vs-post-ln",
    "rmsnorm-vs-layernorm": "rmsnorm-vs-layernorm",
    "swiglu-mlp": "swiglu-mlp",
    "residual-stream": "gpt2-residual-stream",
    "weight-tying": "weight-tying",
    "parallel-block": "parallel-block",
    "encoder-decoder-family": "encoder-decoder-family",
    "transformer-depth-vs-width": "transformer-depth-vs-width",
}


def needs(name):
    preset = PRESETS[name]
    return pytest.mark.skipif(not available(preset), reason=preset + ": data not downloaded")


def cases(names=tuple(PRESETS)):
    return [pytest.param(n, marks=needs(n)) for n in names]


def build(name):
    return presets.get(PRESETS[name]).build()


def texts(anim, t):
    return sorted((x.s, round(float(x.opacity), 3)) for x in anim.scene(float(t)).texts)


def said(anim):
    """Everything the animation's subtitles say."""
    return " | ".join(anim.subtitle_for(k) for k in range(len(anim.PHASES)))


@pytest.mark.parametrize("name", cases())
def test_registered_drawn_and_laid_out(name):
    assert name in M.ANIMATIONS
    anim = build(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.t_done <= 60.0 and 4 <= len(anim.PHASES) <= 5
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0


REDESIGNED = ("pre-ln-vs-post-ln", "rmsnorm-vs-layernorm", "swiglu-mlp", "parallel-block",
              "transformer-depth-vs-width")


def study_of(name):
    return build_study("next_attention", name)


@pytest.mark.parametrize("name", cases(REDESIGNED))
def test_redesigned_study_is_what_is_drawn(name):
    """The catalogue builds the redesigned experiment, and its data are what they say."""
    assert name in SB.STUDIES
    study, e = study_of(name), SB.experiment(name)
    m = study.metadata
    assert study.subtitle.startswith("Real text") and len(study.panels) == 4
    # disjoint text: training windows from the first 80%, held-out from the last 10%
    assert m["dataset"]["character_range"][1] <= m["evaluation_dataset"]["character_range"][0]
    (x, y), (xt, yt) = e["train"], e["test"]
    assert x.shape == (m["training_windows"], m["window_length"])
    assert xt.shape == (SB.TEST_WINDOWS, m["window_length"])
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    np.testing.assert_array_equal(xt[:, 1:], yt[:, :-1])
    train_text, test_text = R.text("train"), R.text("test")
    for window, text in ((x[0], train_text), (xt[0], test_text), (xt[-1], test_text)):
        assert R.decode(window).replace("\ufffd", "?") in text.replace("\ufffd", "?") or 0 in window
    # every variant, every seed: a different trained model, scored once
    held = np.asarray(m["heldout_loss_by_seed"])
    assert held.shape == (len(e["configs"]), len(m["seeds"])) and len(m["seeds"]) >= 3
    assert len({round(float(v), 9) for v in held.ravel()}) == held.size
    np.testing.assert_allclose(m["heldout_loss"], held.mean(1))
    assert len(m["runs"]) == held.size
    for runs, c, row in zip(e["runs"], e["configs"], held):
        for run, value in zip(runs, row):  # noqa: B007
            assert len(run["loss"]) == m["steps"] == run["provenance"]["actual_updates"]
        assert T.loss(runs[0]["P"], xt, yt, c) == pytest.approx(row[0], abs=1e-9)
        assert len({r["provenance"]["initial_parameter_sha256"] for r in runs}) == len(runs)
    assert len({r["provenance"]["final_parameter_sha256"] for rs in e["runs"] for r in rs}) \
        == held.size
    assert np.all(held < np.log(len(R.vocabulary())) - 0.5)       # all beat guessing evenly


@pytest.mark.parametrize("name", cases(REDESIGNED))
def test_mechanisms_are_drawn_from_the_first_seed(name):
    configs, runs, losses, _test = M.trained(name)
    e = SB.experiment(name)
    assert configs == e["configs"]
    assert all(run is rs[0] for run, rs in zip(runs, e["runs"]))
    np.testing.assert_array_equal(losses, e["heldout"][:, 0])


@pytest.mark.parametrize("name", cases())
def test_loop_closes(name):
    anim = build(name)
    assert texts(anim, 0.0) == texts(anim, anim.duration)
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    assert len(first.quads) == len(last.quads) and len(first.lines) == len(last.lines)


@needs("weight-tying")
def test_checkpoints_are_the_studys(name="weight-tying"):
    """The models the mechanisms are drawn from score what the study reported."""
    study = NA.make(name)
    _configs, runs, losses, _test = M.trained(name)
    np.testing.assert_allclose(losses, study.panels[2].data["values"], rtol=0, atol=1e-12)
    traces = np.array([run["loss"] for run in runs])
    np.testing.assert_allclose(traces, study.panels[1].data["ys"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", cases(["pre-ln-vs-post-ln", "rmsnorm-vs-layernorm",
                                         "swiglu-mlp", "parallel-block"]))
def test_block_trace_is_the_models_forward_pass(name):
    configs, runs, _losses, _test = M.trained(name)
    ids, _chars = M.test_window()
    for c, run in zip(configs[:2], runs[:2]):
        trace = M.block_trace(run["P"], ids, c)
        _logits, rec = T.forward(run["P"], ids[None], c, keep=True)
        np.testing.assert_allclose(trace["x"], rec["residual"][0][0], atol=1e-12)
        np.testing.assert_allclose(trace["out"], rec["residual"][1][0], atol=1e-12)
        np.testing.assert_allclose(trace["mid_raw"], trace["x"] + trace["attn"], atol=1e-12)
        if c.activation == "swiglu":
            np.testing.assert_allclose(trace["hidden"], trace["silu"] * trace["gate"], atol=1e-12)
        if c.parallel:                      # the MLP was given the incoming stream, normalised
            v = trace["x"] - trace["x"].mean(-1, keepdims=True)
            v = v / np.sqrt((v ** 2).mean(-1, keepdims=True) + 1e-5)
            np.testing.assert_allclose(
                trace["reads"], v * run["P"]["l0.n2.g"] + run["P"]["l0.n2.b"], atol=1e-10)


# ---------------------------------------------------------------- 13
@needs("pre-ln-vs-post-ln")
def test_pre_post_ln_study():
    """Deep post-LN does not train at the rate pre-LN trains at; shallow post-LN does;
    a warm-up helps deep post-LN."""
    m = study_of("pre-ln-vs-post-ln").metadata
    assert m["depths"] == [2, 12] and m["warmup_updates"] == [0, 0, 100, 0, 0, 100]
    assert m["designs_tried"] and m["learning_rate"] == 0.003
    held = np.asarray(m["heldout_loss_by_seed"]).reshape(2, 3, -1)     # depth, arm, seed
    pre, post, warm = held[1]
    assert post.min() > pre.max() + 0.5                    # every seed, by a wide margin
    assert abs(post.mean() - M.frequency_loss()) < 0.05    # it learned letter frequencies only
    assert warm.mean() < post.mean() - 0.2
    assert warm.min() < post.min() - 0.5                   # at least one seed escapes
    assert np.abs(held[0] - held[0].mean()).max() < 0.05   # at 2 blocks the arms agree
    assert held[1, 0].mean() < held[0, 0].mean() + 0.05    # pre-LN loses nothing with depth
    # training text: the letter-frequency loss is the entropy-like floor of knowing no context
    assert 3.0 < M.frequency_loss() < np.log(len(R.vocabulary()))


@needs("pre-ln-vs-post-ln")
def test_pre_post_ln_numbers():
    anim, study = build("pre-ln-vs-post-ln"), study_of("pre-ln-vs-post-ln")
    held = np.asarray(study.panels[3].data["values"]).reshape(2, 3)
    np.testing.assert_allclose(anim.heldout, held)
    assert "12 blocks: pre-LN %.2f nats; post-LN stuck at %.2f (letter frequencies)" % (
        held[1, 0], held[1, 1]) in anim.subtitle_for(4)
    assert "2 blocks: all three train" in anim.subtitle_for(4)
    assert [anim.stuck(1, arm) for arm in range(3)] == [0, 3, 1]
    assert "with warm-up, trains in 2 of 3 seeds" in anim.subtitle_for(4)
    g = np.asarray(study.panels[0].data["ys"])
    assert "between %.2g and %.2g; post-LN between %.2g and %.2g" % (
        g[0].min(), g[0].max(), g[1].min(), g[1].max()) in anim.subtitle_for(3)
    np.testing.assert_array_equal(anim.grads, g)
    np.testing.assert_allclose(g, SB.initial_gradients().mean(1))
    assert g[1, 0] < g[0, 0] / 10 and "times smaller" in anim.subtitle_for(3)
    # the traces drawn: 12 blocks in panel 1, 2 blocks in panel 2, three arms each
    assert anim.traces.shape == (2, 3, 800 // SB.BLOCK) and anim.steps[-1] == 800
    np.testing.assert_allclose(
        anim.traces[1, 1, 0], SB.experiment("pre-ln-vs-post-ln")["traces"][4, :, :SB.BLOCK].mean())
    # the stream's size through 12 untrained blocks, recomputed from the model
    sizes = M.depth_sizes()
    assert sizes.shape == (2, 13)
    e = SB.experiment("pre-ln-vs-post-ln")
    for row, c in zip(sizes, e["configs"][3:5]):
        assert (c.layers, c.width) == (12, 16)
        _logits, rec = T.forward(T.init(c, 0), e["train"][0][:e["batch"]], c, keep=True)
        assert row[7] == pytest.approx(np.linalg.norm(rec["residual"][7], axis=-1).mean())
    assert sizes[0, 0] == pytest.approx(sizes[1, 0])          # the same embeddings enter both
    # post-LN: a normalised vector of 16 numbers has length sqrt(16), whatever came in
    np.testing.assert_allclose(sizes[1, 1:], 4.0, atol=1e-3)
    assert sizes[0, -1] > 2.5 * sizes[1, -1]
    assert "%.1f to %.1f" % (sizes[0, 0], sizes[0, -1]) in anim.subtitle_for(2)


# ---------------------------------------------------------------- 14, 15, 18: eight seeds
def test_paired_gap_is_the_paired_t_test():
    rng = np.random.default_rng(0)
    base = rng.normal(2.3, 0.02, 8)
    gap, noise, wins, real = M.paired_gap([base, base + 0.02 + rng.normal(0, 0.004, 8)])
    assert real and wins == 0 and gap == pytest.approx(0.02, abs=0.005)
    assert noise == pytest.approx(np.ptp(base), abs=0.02)
    gap, _noise, wins, real = M.paired_gap([base, base + rng.normal(0, 0.02, 8)])
    assert not real and 0 < wins < 8


@pytest.mark.parametrize("name", cases(["rmsnorm-vs-layernorm", "swiglu-mlp", "parallel-block"]))
def test_paired_studies_report_every_seed(name):
    anim, study = build(name), study_of(name)
    m = study.metadata
    held = np.asarray(m["heldout_loss_by_seed"])
    assert held.shape == (2, 8) and m["steps"] == 2000
    np.testing.assert_array_equal(anim.by_seed, held)
    np.testing.assert_array_equal(study.panels[0].data["values"], held)
    diff = held[1] - held[0]
    assert m["paired_difference_mean"] == pytest.approx(diff.mean())
    assert m["seeds_second_is_better"] == int((diff < 0).sum()) == anim.wins
    assert anim.mean_gap == pytest.approx(diff.mean())
    a, b = anim.NAMES
    last = len(anim.PHASES) - 1
    assert "mean of 8 runs: %s %.3f nats, %s %.3f" % (a, held[0].mean(), b, held[1].mean()) \
        in anim.subtitle_for(last)
    assert ("a real difference" in anim.subtitle_for(last)) == anim.real
    assert ("within the noise" in anim.subtitle_for(last)) == (not anim.real)
    assert "each trained 8 times" in anim.subtitle_for(last - 1)
    assert anim.params == [int(v) for v in study.panels[3].data["values"]]
    assert anim.params[0] == anim.params[1]                 # equal learned numbers


@needs("rmsnorm-vs-layernorm")
def test_rmsnorm_is_within_the_noise():
    """Dropping the mean changes the loss by less than seeds do."""
    anim = build("rmsnorm-vs-layernorm")
    assert not anim.real and abs(anim.mean_gap) < 0.015 < anim.noise
    assert 2 <= anim.wins <= 6


@needs("swiglu-mlp")
def test_swiglu_is_not_better_here():
    """At equal weights and this size the gated MLP does not beat GELU."""
    anim = build("swiglu-mlp")
    assert not anim.real and abs(anim.mean_gap) < 0.02 < anim.noise
    assert anim.heldout[1] > anim.heldout[0] - 0.005


@needs("parallel-block")
def test_parallel_block_costs_loss():
    """The parallel block is worse than the sequential one beyond the seeds' noise."""
    anim = build("parallel-block")
    assert anim.real and anim.mean_gap > 0.008 and anim.wins <= 2
    assert "Sequential is better in %d of the 8 seeds" % (8 - anim.wins) in anim.subtitle_for(3)


@needs("rmsnorm-vs-layernorm")
def test_rmsnorm_numbers():
    anim = build("rmsnorm-vs-layernorm")
    # the arithmetic drawn is the model's own normalisation (with unit gain, no shift)
    one = {"n.g": T.variable(np.ones(16)), "n.b": T.variable(np.zeros(16))}
    for norm, mine in (("layer", anim.ln), ("rms", anim.rn)):
        theirs = T._norm(T.variable(anim.v[None]), one, "n", T.Config(norm=norm)).v[0]
        np.testing.assert_allclose(mine, theirs, atol=1e-12)
    assert abs(anim.ln.mean()) < 1e-12 and anim.ln.std() == pytest.approx(1, abs=1e-4)
    assert np.sqrt((anim.rn ** 2).mean()) == pytest.approx(1, abs=1e-4)
    assert anim.gap == pytest.approx(np.abs(anim.ln - anim.rn).max())
    assert "%.2f" % anim.v.mean() in anim.subtitle_for(0)


# ---------------------------------------------------------------- 15
@needs("swiglu-mlp")
def test_swiglu_numbers():
    anim = build("swiglu-mlp")
    assert anim.mlp_params == [16 * 48 * 2, 16 * 32 * 3] and len(set(anim.mlp_params)) == 1
    assert (len(anim.pre), len(anim.silu)) == (48, 32)
    np.testing.assert_allclose(anim.product, anim.silu * anim.gate, atol=1e-12)
    np.testing.assert_allclose(anim.after, T.gelu(T.variable(anim.pre)).v, atol=1e-12)
    j = anim.star
    assert "%.2f x %.2f = %.2f" % (anim.silu[j], anim.gate[j], anim.product[j]) \
        in anim.subtitle_for(1)


# ---------------------------------------------------------------- 16
@needs("residual-stream")
def test_residual_stream_numbers():
    anim, study = build("residual-stream"), NA.make("residual-stream")
    attn, mlp = np.asarray(study.panels[0].data["ys"])
    lens = np.asarray(study.panels[1].data["ys"])[0]
    assert "largest in layer 12 (%.0f and %.0f)" % (attn[-1], mlp[-1]) in anim.subtitle_for(1)
    assert "falls from %.0f%% to %.0f%%" % (100 * lens[22], 100 * lens[24]) \
        in anim.subtitle_for(3)
    t = M.stream_trace()
    # the read-outs computed here agree with the study's curve for the final answer
    assert repr(t["answer"]) == study.panels[1].data["labels"][0]
    assert len(t["top"]) == 25 and len(t["length"]) == 13
    for p in range(25):
        if t["top"][p] == t["answer"]:
            assert t["top_p"][p] == pytest.approx(lens[p])
        else:
            assert t["top_p"][p] >= lens[p]
    assert t["final"][0][0] == t["answer"] and t["final"][0][1] == pytest.approx(lens[24])
    # each layer's two writes are what moves the stream from one length to the next
    assert np.all(np.abs(np.diff(t["length"])) <= attn + mlp + 1e-9)
    assert anim.tokens_[-1] == "row" and anim.answer == '","'


# ---------------------------------------------------------------- 17
@needs("weight-tying")
def test_weight_tying_numbers():
    anim, study = build("weight-tying"), NA.make("weight-tying")
    held = study.metadata["heldout_loss"]
    assert "untied model scores %.2f nats and the tied one %.2f" % tuple(held) \
        in anim.subtitle_for(4)
    assert anim.saved_gpt2 == 50257 * 768 and "38,597,376" in anim.subtitle_for(1)
    assert anim.params[0] - anim.params[1] == 66 * 16
    assert anim.tables[1][0] is anim.tables[1][1]             # tied: one table at both ends
    whole, g_in, g_out, x = M.tied_gradient()
    np.testing.assert_allclose(whole, g_in + g_out, atol=1e-12)
    read = np.zeros(66, bool)
    read[np.unique(x)] = True
    np.testing.assert_array_equal(np.abs(g_in).sum(1) > 0, read)   # only the rows looked up
    assert anim.rows_in == int(read.sum()) and anim.rows_out == 66
    # what each model is drawn answering is its own most likely next character
    configs, runs, _losses, _test = M.trained("weight-tying")
    for (k, p, logits), c, run in zip(anim.said, configs, runs):
        assert k == int(T.forward(run["P"], anim.ids[None], c)[0, anim.AT].argmax())
        assert 0 < p <= 1 and logits.shape == (66,)


# ---------------------------------------------------------------- 18
@needs("parallel-block")
def test_parallel_block_numbers():
    anim = build("parallel-block")
    seq, par = anim.flow
    assert "%.1f to %.1f" % (M.size(par["x"][anim.at]), M.size(par["out"][anim.at])) \
        in anim.subtitle_for(1)
    # sequential: the MLP's write is added to what attention left; parallel: both add to x
    np.testing.assert_allclose(seq["out"], seq["x"] + seq["attn"] + seq["mlp"], atol=1e-12)
    np.testing.assert_allclose(par["out"], par["x"] + par["attn"] + par["mlp"], atol=1e-12)


# ---------------------------------------------------------------- 19
@needs("encoder-decoder-family")
def test_family_numbers():
    anim, study = build("encoder-decoder-family"), NA.make("encoder-decoder-family")
    held = study.panels[3].data["values"]
    assert "encoder only %.2f nats, decoder only %.2f, encoder-decoder %.2f" % tuple(held) \
        in anim.subtitle_for(4)
    source, shifted, target, answers = M.family_answers()
    # the checkpoints replay the study's joint accuracy and the masks are the models'
    assert "Joint target accuracy: %.1f%%" % (100 * anim.accuracy[2]) in study.panels[3].notes
    assert (answers[2] == target).mean() == pytest.approx(0.3046875)
    np.testing.assert_array_equal(shifted[:, 1:], target[:, :-1])
    assert (source[:, 1::3] == 0).all() and (source[:, 0::3] == target[:, 0::3]).all()
    enc, causal, _dec = M.family_configs()
    np.testing.assert_array_equal(anim.masks[0], T.attention_mask(12, enc))
    np.testing.assert_array_equal(anim.masks[1], T.attention_mask(12, causal))
    both = M.family_mask()
    assert both.shape == (12, 12) and both[:6, :6].all() and not both[:6, 6:].any()
    assert both[6:, :6].all() and np.array_equal(both[6:, 6:], np.tril(np.ones((6, 6))))
    assert anim.source.count("?") == 2 and len(anim.said) == 3
    assert [sum(r) for r in anim.right] == [
        int((answers[m][anim.ROW] == target[anim.ROW]).sum()) for m in range(3)]


# ---------------------------------------------------------------- 20
@needs("transformer-depth-vs-width")
def test_depth_width_study():
    """Equal weights, text seen once: nothing is memorised, one and two blocks cannot be
    told apart, and eight narrow blocks are worse in every seed."""
    study = study_of("transformer-depth-vs-width")
    m = study.metadata
    params = m["parameters"]
    assert params == [62848, 62880, 62912, 62736] and max(params) < 1.005 * min(params)
    assert [(s["blocks"], s["width"]) for s in m["shapes"]] == [(1, 64), (2, 48), (4, 32), (8, 24)]
    # about one pass: the characters trained on do not exceed the training text
    assert m["steps"] * m["batch"] == m["training_windows"] == 32000
    assert m["steps"] * m["batch"] * m["window_length"] < len(R.text("train"))
    held = np.asarray(m["heldout_loss_by_seed"])
    fitted = np.asarray(m["final_training_loss"])
    assert np.all(held.mean(1) - fitted < 0.25) and np.all(held.mean(1) > fitted)
    assert np.all(held < m["uniform_guess"] - 1.8)          # the first study's were above it
    one, two, _four, eight = held
    assert abs(one.mean() - two.mean()) < 0.03 and one.min() < two.max() and two.min() < one.max()
    assert eight.min() > max(one.max(), two.max()) + 0.02
    # the loss by context: all shapes gain from the first few characters, none from more depth
    context = np.asarray(study.panels[3].data["ys"])
    assert context.shape == (4, 16)
    assert np.all(context[:, 0] - context[:, 3] > 0.4)
    late = context[:, 8:].mean(1)
    assert late[0] - late[1:].min() < 0.03
    assert "No two-hop composition task" in m["not_measured"]


@needs("transformer-depth-vs-width")
def test_depth_width_numbers():
    anim, study = build("transformer-depth-vs-width"), study_of("transformer-depth-vs-width")
    held = study.metadata["heldout_loss"]
    assert "1 block %.2f, 2 blocks %.2f, 4 blocks %.2f, 8 blocks %.2f" % tuple(held) \
        in anim.subtitle_for(3)
    assert "1 and 2 overlap across seeds; 4 and 8 are worse in every seed" in anim.subtitle_for(3)
    assert "past 2 blocks, depth did not pay here" in anim.subtitle_for(3)
    assert not anim.deeper_reads_further
    assert "depth did not make more of a long context" in anim.subtitle_for(4)
    assert anim.params == [62848, 62880, 62912, 62736]
    assert anim.guess == pytest.approx(np.log(66))
    np.testing.assert_allclose(
        anim.fitted,
        SB.experiment("transformer-depth-vs-width")["traces"][:, :, -100:].mean((1, 2)))
    maps = M.layer_attention()
    assert [m.shape for m in maps] == [(1, 12, 12), (2, 12, 12), (4, 12, 12), (8, 12, 12)]
    for m in maps:
        np.testing.assert_allclose(m.sum(-1), 1.0, atol=1e-9)       # each reader's shares
        assert np.allclose(np.triu(m, 1), 0.0)                      # and none on later characters

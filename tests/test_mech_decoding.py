"""Animations 29-34 (decoding and serving): layout, the numbers on screen
against their studies, the traces the mechanisms are drawn from, and what the
redesigned studies (30-33, :mod:`voxanim.study_decoding`) now claim."""

import numpy as np
import pytest
from conftest import data_marks

from voxanim import mech_decoding as M
from voxanim import presets, qa
from voxanim import study_decoding as SD

PRESETS = {
    "beam-search": "gpt2-beam-search",
    "speculative-decoding": "gpt2-speculative-decoding",
    "repetition-and-penalties": "gpt2-repetition-and-penalties",
    "constrained-decoding": "gpt2-constrained-decoding",
    "contrastive-decoding": "gpt2-contrastive-decoding",
    "batching-and-throughput": "batching-and-throughput",
}


def needs(name):
    """``name`` as a parameter carrying the marks for the data its preset needs."""
    return pytest.param(name, marks=data_marks(PRESETS[name]))


def gpt2_test(name):
    return pytest.mark.parametrize("name", [needs(name)])


_built = {}


def build(name):
    if name not in _built:
        _built[name] = presets.get(PRESETS[name]).build()
    return _built[name]


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


@pytest.mark.parametrize("name", [needs(n) for n in PRESETS])
def test_drawn_by_this_module_and_lays_out_cleanly(name):
    anim = build(name)
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0
    for line in subtitles(anim):
        assert line and "nan" not in line


@pytest.mark.parametrize("name", [needs(n) for n in PRESETS])
def test_loop_closes(name):
    anim = build(name)
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    key = lambda sc: sorted((t.s, round(t.x, 4), round(t.y, 4), round(t.opacity, 3))  # noqa: E731
                            for t in sc.texts)
    assert key(first) == key(last)
    assert len(first.lines) == len(last.lines) == 0


# ---------------------------------------------------------------------- beam search
@gpt2_test("beam-search")
def test_beam_tree_is_the_studys_search(name):
    anim = build(name)
    study, t = anim.study, anim.trace
    widths = study.metadata["settings"]["widths"]
    final = study.panels[2].data["values"]
    for w, score in zip(widths, final):
        assert t["score%d" % w][-1, 0] == pytest.approx(score, abs=1e-9)
    # the kept scores of the widest beam are the study's matrix, step by step
    np.testing.assert_allclose(t["score8"][:, :8], study.panels[1].data["values"], atol=1e-9)
    # every candidate extends a kept hypothesis by one token: its score is the parent's
    # plus a log-probability (so never higher), and the kept ones are the best
    tok, score, parent = t["tok4"], t["score4"], t["parent4"]
    assert (tok[0, :4] >= 0).all() and (tok[0, 4:] < 0).all() and (tok[1:] >= 0).all()
    for k in range(1, M.BEAM_STEPS):
        assert (score[k] <= score[k - 1, parent[k]] + 1e-12).all()
        assert (np.diff(score[k]) <= 1e-12).all()
        assert set(parent[k]) <= set(range(4))
    # the greedy fan: the likeliest token is the one greedy keeps, and its log is the step
    assert [int(i) for i in t["alt_tok"][:, 0]] == [int(i) for i in t["best1"]]
    np.testing.assert_allclose(np.cumsum(np.log(t["alt_p"][:, 0])),
                               study.panels[3].data["ys"][0], atol=1e-6)


@gpt2_test("beam-search")
def test_beam_numbers_on_screen(name):
    anim = build(name)
    final = [float(v) for v in anim.study.panels[2].data["values"]]
    lines = subtitles(anim)
    assert "%.2f" % final[0] in lines[0] and "%.2f" % final[0] in lines[3]
    assert "%.2f" % final[2] in lines[3]
    assert "%.1f times" % np.exp(final[2] - final[0]) in lines[3]
    assert " ".join(anim.texts[1]) in lines[0]
    assert " ".join(anim.texts[4]) == anim.study.panels[3].notes[2]
    # the two paths: the winner is kept at every step, greedy's is cut exactly once, at the end
    assert all(c is not None and c < 4 for c in anim.best_path)
    assert all(c < 4 for c in anim.greedy_path[:-1]) and anim.greedy_path[-1] >= 4
    assert "ranked %d of 16" % (anim.greedy_path[-1] + 1) in lines[2]


# ---------------------------------------------------------------------- the redesigned studies
def passage(ids):
    return SD._tok().decode([int(i) for i in ids])


@gpt2_test("speculative-decoding")
def test_stream_reads_gpt2_as_the_hooks_do(name):
    """The incremental forward pass the redesigned studies use gives GPT-2's logits: all at
    once, token by token, after forgetting a suffix, and at the four-layer exit."""
    from voxanim import gpt2_hooks
    ids = [int(i) for i in SD.heldout_prompts(2, 20)[1]]
    want = gpt2_hooks.forward(ids)["logits"]
    stream = SD.Stream()
    got = np.vstack([SD.unembed(stream.feed(ids[:12])), SD.unembed(stream.feed(ids[12:13])),
                     SD.unembed(stream.feed(ids[13:]))])
    np.testing.assert_allclose(got, want, atol=2e-3)
    stream.n = 9
    np.testing.assert_allclose(SD.unembed(stream.feed(ids[9:])), want[9:], atol=2e-3)
    stream.n = 0
    taps = stream.feed(ids, taps=(4,))
    np.testing.assert_allclose(SD.unembed(taps[4]), gpt2_hooks.forward(ids, n_layers=4)["logits"],
                               atol=2e-3)
    np.testing.assert_allclose(SD.probs(taps[12]).sum(1), 1.0)


@gpt2_test("speculative-decoding")
def test_heldout_passages_are_test_text_only(name):
    from voxanim import realdata
    train, test = realdata.text("train"), realdata.text("test")
    for count, length in ((SD.SPEC_PROMPTS, SD.SPEC_PROMPT_TOKENS), (SD.PROMPTS, SD.PROMPT_TOKENS)):
        prompts = SD.heldout_prompts(count, length)
        assert prompts.shape == (count, length)
        texts = [passage(p) for p in prompts]
        assert len(set(texts)) == count
        assert all(t in test and t not in train for t in texts)


# ---------------------------------------------------------------------- speculative decoding
def test_keep_or_replace_leaves_the_targets_distribution():
    rng = np.random.default_rng(3)
    p = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    q = np.array([0.1, 0.6, 0.0, 0.1, 0.2])
    np.testing.assert_allclose(SD.speculative_distribution(p, q), p, atol=1e-15)
    counts = np.zeros(5)
    for _ in range(20000):
        guess = int(rng.choice(5, p=q))
        kept, other = SD.speculative_step(p, q, guess, rng)
        counts[guess if kept else other] += 1
    np.testing.assert_allclose(counts / counts.sum(), p, atol=0.012)
    assert counts[4] == 0                                     # GPT-2 gives it no probability


def test_the_selection_rule_arithmetic():
    assert SD.tokens_per_pass(0.5, 3) == pytest.approx(1 + 0.5 + 0.25 + 0.125)
    assert SD.predicted_speedup(0.5, 0.0, 3) == pytest.approx(1.875)
    assert SD.predicted_speedup(0.5, 0.5, 2) == pytest.approx(1.75 / 2)
    assert SD.draft_cost("bigram") == 0.0
    assert SD.draft_cost("exit", 12) == pytest.approx(1.0)
    assert SD.draft_cost("tuned", 4) > SD.draft_cost("exit", 4) > SD.draft_cost("exit", 0) > 0.3


@gpt2_test("speculative-decoding")
def test_speculative_study_saves_passes_with_the_chosen_draft(name):
    """The redesign's claim: a draft chosen for being cheap and agreeing with GPT-2 saves
    GPT-2 passes on every seed, and more than the first design's draft does."""
    d = SD.speculative_data()
    study = build(name).study
    tokens = SD.SPEC_NEW
    assert d["outputs"].shape == (2, SD.SPEC_PROMPTS, SD.SPEC_SEEDS, tokens)
    # accounting: a pass writes the guesses it keeps and one token of GPT-2's own
    np.testing.assert_array_equal(d["passes"] + d["kept"], tokens)
    assert (d["kept"] <= d["checked"]).all() and (d["checked"] <= d["drafts"]).all()
    per_seed = d["passes"].sum(1) / (SD.SPEC_PROMPTS * tokens)            # arm x seed
    assert per_seed[0].max() < 1.0 - 0.1                 # the chosen draft, every seed
    assert per_seed[0].max() < per_seed[1].min() - 0.05  # ... against the first design
    kept = d["kept"].sum((1, 2)) / d["checked"].sum((1, 2))
    assert kept[0] > kept[1] + 0.05
    np.testing.assert_allclose(study.panels[2].data["values"][1:], per_seed.mean(1))
    np.testing.assert_allclose(study.panels[2].data["per_seed"], per_seed)
    # the draft was chosen by the stated rule, from agreement measured on validation text
    speed = np.asarray([[SD.predicted_speedup(a, c, b) for b in SD.BLOCKS]
                        for a, c in zip(d["alpha"], d["cost"])])
    np.testing.assert_allclose(speed, d["speed"])
    best = np.unravel_index(speed.argmax(), speed.shape)
    assert int(d["chosen"]) == best[0] and int(d["chosen_block"]) == SD.BLOCKS[best[1]]
    assert study.metadata["settings"]["block"] == int(d["chosen_block"])
    n = len(SD.EXIT_LAYERS)
    assert (d["alpha"][n:2 * n] > d["alpha"][:n] + 0.1).all()      # a trained read-out agrees more
    control = SD.EXIT_LAYERS.index(SD.CONTROL[1])
    assert d["speed"][control].max() < 1.0 < d["speed"].max()      # the first design: slower
    assert len(study.metadata["designs_tried"]) >= 2
    # exactness: the algebra on every checked guess, and by simulation at eight positions
    assert float(d["identity_error"]) < 1e-9
    assert (np.abs(d["mc_spec"] - d["mc_direct"]) < 0.25 * d["mc_direct"] + 0.002).all()
    # the decoded text starts from held-out passages, and the two methods are separate draws
    assert (d["outputs"] < SD.V).all() and (d["plain_out"] < SD.V).all()
    assert (d["seconds"] > 0).all() and (d["plain_seconds"] > 0).all()


@gpt2_test("speculative-decoding")
def test_speculative_trace_is_the_studys_run(name):
    anim = build(name)
    d, t = anim.d, anim.trace
    assert len(t["start"]) == int(d["passes"][0, 0, 0]) == anim.n_passes
    assert anim.accepted == int(d["kept"][0, 0, 0]) and anim.checked == int(d["checked"][0, 0, 0])
    # a round writes its accepted guesses and then GPT-2's own token
    written = (t["state"] == 1).sum(1) + 1
    assert np.array_equal(np.cumsum(written)[:-1], t["start"][1:])
    assert written.sum() == len(anim.output) == SD.SPEC_NEW
    out = []
    for r in range(len(written)):
        out += [int(v) for v in t["prop"][r][t["state"][r] == 1]] + [int(t["extra"][r])]
    assert out == anim.output
    for row, size in zip(t["state"], anim.size):
        seen = list(row[:size])
        assert (row[size:] == -1).all() and size <= anim.block
        if 2 in seen:                                  # nothing is checked after a rejection
            assert all(s == 0 for s in seen[seen.index(2) + 1:])
            assert all(s == 1 for s in seen[:seen.index(2)])
    drafted = t["prop"] >= 0
    assert (t["q"][drafted] > 0).all() and (t["p"][drafted] >= 0).all()
    assert np.minimum(t["id_p"], t["id_q"]).sum() <= float(t["overlap"]) + 1e-9
    assert 0 < float(t["overlap"]) < 1
    np.testing.assert_allclose(t["id_freq"], t["id_p"], atol=0.01)   # the simulated draws


@gpt2_test("speculative-decoding")
def test_speculative_numbers_on_screen(name):
    anim = build(name)
    study = anim.study
    lines = subtitles(anim)
    assert "%d of %d checked guesses were kept" % (anim.accepted, anim.checked) in lines[2]
    assert "%d passes for %d tokens" % (anim.n_passes, anim.n_tokens) in lines[2]
    results = study.metadata["results"]
    assert "%.2f GPT-2 passes per token" % results["chosen"]["passes_per_token"] in lines[5]
    assert "%.2f with the first design" % results["control"]["passes_per_token"] in lines[5]
    seconds = study.metadata["timing_seconds"]
    assert "%.1f s plain" % seconds["baseline"] in lines[5]
    assert "%.1f s speculative" % seconds["speculative"] in lines[5]
    assert ("slower" in lines[5]) == (seconds["speculative"] > seconds["baseline"])
    assert anim.draft_name in lines[4] and anim.draft_name == anim.names[anim.chosen]
    assert anim.chosen in anim.shortlist and len(anim.shortlist) <= 6
    assert anim.passes[0] == 1.0


# ---------------------------------------------------------------------- repetition
@gpt2_test("repetition-and-penalties")
def test_penalty_trace_is_the_studys_reshaping(name):
    anim = build(name)
    study, t = anim.study, anim.trace
    np.testing.assert_allclose(t["prob"], study.panels[0].data["values"], atol=1e-12)
    for r in range(len(M.RULES)):
        assert t["roll_p"][r, 0] == pytest.approx(float(np.max(t["prob"][r])), abs=1e-9)
    assert [M.show(i) for i in t["top"]] == list(study.panels[0].data["labels"])
    # with no rule GPT-2 repeats the prompt's three tokens; no rollout under the
    # trigram rule ever completes a three-token run it has already written
    ids = M._tok().encode(M.PENALTY_PROMPT)
    assert [int(i) for i in t["roll"][0][:6]] == ids[-3:] * 2
    assert SD.repeats_trigram(ids, t["roll"][0]) and not SD.repeats_trigram(ids, t["roll"][3])


def test_reshape_logits_rules():
    logits = np.array([-2.0, 3.0, -1.0, 0.5, 2.0])
    seq = [1, 0, 1, 2, 0, 1]
    np.testing.assert_allclose(SD.reshape_logits(logits, seq, "repetition"),
                               [-2.6, 3.0 / 1.3, -1.3, 0.5, 2.0])
    np.testing.assert_allclose(SD.reshape_logits(logits, seq, "frequency"),
                               [-2.0 - 1.4, 3.0 - 2.1, -1.0 - 0.7, 0.5, 2.0])
    banned = SD.reshape_logits(logits, seq, "trigram")        # (0, 1) was followed by 2
    assert np.isneginf(banned[2]) and np.isfinite(np.delete(banned, 2)).all()
    assert SD.reshape_logits(logits, seq, "none") is not None
    np.testing.assert_array_equal(SD.reshape_logits(logits, seq, "none"), logits)


def test_loop_measures():
    assert SD.repeats_trigram([1, 2, 3, 4], [1, 2, 3])            # copies the prompt
    assert SD.repeats_trigram([1, 2], [3, 4, 5, 3, 4, 5])         # copies itself
    assert not SD.repeats_trigram([1, 2, 3, 4], [1, 2, 4, 3, 2, 1])
    assert SD.distinct_fraction([7, 7, 8, 9]) == 0.75


@gpt2_test("repetition-and-penalties")
def test_penalties_break_loops_across_passages(name):
    """Greedy GPT-2 loops on most held-out passages; each rule loops less often, and
    every rule makes GPT-2 write tokens it finds less likely."""
    d = SD.penalty_data()
    study = build(name).study
    loops, own = d["loops"], d["own"]
    assert loops.shape == (len(SD.RULES), SD.PROMPTS)
    for r, rule in enumerate(SD.RULES):                 # the flags are the rollouts' own  # noqa: B007, E501
        for pi in range(SD.PROMPTS):
            tokens = d["across_roll"][r, pi]
            assert loops[r, pi] == SD.repeats_trigram(d["prompts"][pi], tokens)
            assert d["distinct"][r, pi] == SD.distinct_fraction(tokens)
    rate = loops.mean(1)
    assert rate[0] > 0.5                                 # the failure is common, not one prompt's
    assert rate[1] < rate[0] - 0.3 and rate[2] < rate[0] - 0.1
    assert rate[3] == 0.0                                # by construction
    assert d["distinct"].mean(1)[1:].min() > d["distinct"][0].mean() + 0.05
    assert own.mean(1)[1:].max() < own[0].mean() - 0.2   # the cost
    np.testing.assert_allclose(study.panels[1].data["values"], rate)
    np.testing.assert_allclose(study.panels[3].data["values"], own.mean(1))


@gpt2_test("repetition-and-penalties")
def test_penalty_numbers_on_screen(name):
    anim = build(name)
    p = anim.probs
    lines = subtitles(anim)
    assert "from %.2f to %.2f" % (p[0][0], p[1][0]) in lines[1]
    assert "from %.2f to %.2f" % (p[0][0], p[2][0]) in lines[2]
    assert ("still the likeliest" in lines[2]) == (int(np.argmax(p[2])) == 0)
    assert "leads at %.2f" % p[3].max() in lines[3]
    # the arithmetic shown for the looping token is the rule applied to its real logit
    top = int(anim.trace["top"][0])
    assert anim.count == anim.ids.count(top)
    assert anim.roll[1][0] == anim.roll[3][0]                # "the other two rules begin ..."
    loops = SD.penalty_data()["loops"].sum(1)
    assert "with no rule %d continuations" % loops[0] in lines[5]
    assert "frequency penalty %d, repetition penalty %d" % (loops[2], loops[1]) in lines[5]


# ---------------------------------------------------------------------- constrained decoding
@gpt2_test("constrained-decoding")
def test_grammar_trace_is_the_studys_run(name):
    anim = build(name)
    study, t = anim.study, anim.trace
    np.testing.assert_allclose(t["mass"], study.panels[1].data["values"], atol=1e-9)
    assert [M.show(i) for i in t["choice"]] == list(study.panels[3].data["labels"])
    assert "".join(anim.emitted) == anim.output
    # the mask drawn: ten digits where a digit is due, the comma alone otherwise
    assert anim.allowed.shape == (5, 11)
    for kind, row, pick in zip(M.GRAMMAR, anim.allowed, anim.pick):
        assert row.tolist() == ([1.0] * 10 + [0.0] if kind == "digit" else [0.0] * 10 + [1.0])
        assert row[pick] == 1.0
    np.testing.assert_allclose(anim.rescaled.sum(1), 1.0)
    assert [anim.COLUMNS[k] for k in anim.pick] == anim.emitted
    # the digits GPT-2 scores are the tokens "0".."9"
    assert [M._tok().decode([int(i)]) for i in t["digits"]] == [str(d) for d in range(10)]
    assert (t["want_p"][:, 0] >= t["digit_p"].max(1) - 1e-12).all()


@gpt2_test("constrained-decoding")
def test_grammar_holds_across_passages(name):
    """After every held-out passage, with the instruction or without, the masked output
    has the form; GPT-2 left alone writes it less often."""
    d = SD.grammar_data()
    study = build(name).study
    assert d["across_mass"].shape == (2, SD.PROMPTS, len(SD.GRAMMAR))
    assert ((d["across_mass"] > 0) & (d["across_mass"] <= 1)).all()
    for c in range(2):
        for pi in range(SD.PROMPTS):
            assert SD.FORM.fullmatch(passage(d["across_choice"][c, pi]))
            assert set(d["across_choice"][c, pi]) <= set(d["digits"]) | {int(d["comma"])}
    assert d["across_valid"].all()
    assert d["across_free_valid"].mean() < 1.0 - 0.5
    # asking for digits moves probability onto them: more survives the mask at the first step
    assert np.median(d["across_mass"][1, :, 0]) > 5 * np.median(d["across_mass"][0, :, 0])
    np.testing.assert_allclose(study.panels[2].data["values"], d["across_mass"][1])
    np.testing.assert_allclose(study.panels[2].data["alone"], d["across_mass"][0])


@gpt2_test("constrained-decoding")
def test_grammar_numbers_on_screen(name):
    anim = build(name)
    mass = [float(v) for v in anim.study.panels[1].data["values"]]
    lines = subtitles(anim)
    assert M.pct(mass[0]) in lines[1]
    assert all(M.pct(m) in lines[3] for m in mass)
    assert anim.output in lines[2]
    assert ("no digit" in lines[3]) == (not any(w.isdigit() for w in anim.free))
    d = SD.grammar_data()
    assert "in %d of %d runs, masked in all %d" % (
        d["across_free_valid"].sum(), d["across_valid"].size, d["across_valid"].sum()) in lines[4]
    assert M.pct(np.median(d["across_mass"][1, :, 2])) + " at the comma" in lines[4]


# ---------------------------------------------------------------------- contrastive decoding
@gpt2_test("contrastive-decoding")
def test_contrast_trace_is_the_studys_contrast(name):
    anim = build(name)
    study, t = anim.study, anim.trace
    labels = list(study.panels[0].data["labels"])
    assert [M.show(i) for i in t["top_tok"]] == labels
    kept = int(study.panels[2].notes[0].split(":")[1])
    assert len(t["keep_tok"]) == kept
    for j, token in enumerate(t["keep_tok"]):
        i = labels.index(M.show(token))
        assert t["keep_ep"][j] == pytest.approx(study.panels[0].data["values"][i], abs=1e-12)
        assert t["keep_ap"][j] == pytest.approx(study.panels[1].data["values"][i], abs=1e-12)
        assert t["keep_score"][j] == pytest.approx(study.panels[2].data["values"][i], abs=1e-9)
        assert t["keep_p"][j] == pytest.approx(study.panels[2].data["contrast"][i], abs=1e-12)
    assert t["keep_p"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (t["keep_ep"] >= float(t["threshold"])).all()
    # unfiltered, the largest differences are tokens the expert all but rules out
    assert (t["raw_score"] > t["keep_score"].max()).all()
    assert (t["raw_ep"] < float(t["threshold"])).all()
    assert (np.diff(t["am_ap"]) <= 0).all()


def test_contrast_rule():
    ep = np.array([[0.5, 0.3, 0.15, 0.04, 0.01]])
    ap = np.array([[0.6, 0.1, 0.01, 0.0001, 0.2899]])
    plausible, score, pick = SD.contrast(ep, ap)
    assert plausible.tolist() == [[True, True, True, False, False]]
    np.testing.assert_allclose(score[0, :3], np.log([0.5 / 0.6, 3.0, 15.0]))
    assert pick.tolist() == [2]            # not token 3: a larger difference, but implausible


@gpt2_test("contrastive-decoding")
def test_contrast_across_passages(name):
    """What the redesign measured over held-out passages: the contrast usually picks a
    different token than the expert, and names the real next token less often."""
    d = SD.contrast_data()
    s = SD.contrast_summary(d)
    study = build(name).study
    n = s["positions"]
    assert n == SD.PROMPTS * (SD.PROMPT_TOKENS - 9) == d["differ"].size
    assert not (~d["differ"] & (d["hit_expert"] != d["hit_contrast"])).any()   # same pick, same hit
    assert not (d["differ"] & d["hit_expert"] & d["hit_contrast"]).any()
    assert (d["n_kept"] >= 1).all() and not d["differ"][d["n_kept"] == 1].any()
    assert s["differ"] > 0.5 * n
    assert s["hit_expert"] > s["hit_contrast"] + 0.03 * n
    assert s["expert_right"] > s["contrast_right"]
    assert s["own"][0] > s["own"][1] + 0.2             # greedy is the likelier text, as it must be
    for m in range(2):
        for pi in range(SD.PROMPTS):
            assert d["loops"][m, pi] == SD.repeats_trigram(d["prompts"][pi],
                                                           d["across_roll"][m, pi])
    values = study.panels[3].data["values"]
    np.testing.assert_allclose(values, [s["differ"] / n, s["hit_expert"] / n, s["hit_contrast"] / n,
                                        s["loops"][0] / SD.PROMPTS, s["loops"][1] / SD.PROMPTS])
    assert study.metadata["results"] == s


@gpt2_test("contrastive-decoding")
def test_contrast_numbers_on_screen(name):
    anim = build(name)
    study = anim.study
    lines = subtitles(anim)
    ep, score = (np.asarray(study.panels[k].data["values"], float) for k in (0, 2))
    contrast = np.asarray(study.panels[2].data["contrast"], float)
    assert "%s (%.2f)" % (anim.labels[0], ep[0]) in lines[0]
    assert "scores %+.1f" % score.max() in lines[1]
    assert "%d remain" % anim.kept in lines[2]
    assert "%s gets %.2f" % (anim.selected, contrast.max()) in lines[3]
    assert anim.labels[int(np.argmax(contrast))] == anim.selected
    assert sum(ep >= anim.threshold) == anim.kept
    s = study.metadata["results"]
    assert "differ at %d of %d positions" % (s["differ"], s["positions"]) in lines[4]
    assert "%d expert and %d contrast continuations loop" % tuple(s["loops"]) in lines[4]


# ---------------------------------------------------------------------- batching
def test_static_schedule():
    grid = M.static_schedule(np.array([3, 1, 2, 2, 4]), slots=2)
    assert grid.tolist() == [[0, 0, 0, 2, 2, 4, 4, 4, 4], [1, -1, -1, 3, 3, -1, -1, -1, -1]]


@pytest.mark.parametrize("name", [needs("batching-and-throughput")])
def test_batching_schedules_and_counts(name):
    anim = build(name)
    study = anim.study
    assert anim.durations.sum() == (anim.slots >= 0).sum() == (anim.static >= 0).sum()
    assert anim.static.shape[1] == sum(
        anim.durations[b:b + 4].max() for b in range(0, len(anim.durations), 4))
    for grid in (anim.slots, anim.static):                # each request runs once, unbroken
        for i, d in enumerate(anim.durations):
            r, c = np.nonzero(grid == i)
            assert len(set(r)) == 1 and c.tolist() == list(range(c[0], c[0] + d))
    assert anim.done_at[0] < anim.done_at[1] == anim.static.shape[1]
    for k, b in enumerate(anim.batches):                  # the bars are the study's count
        weights, data, flops = anim.bytes_flops(b)
        assert flops / (weights + data) == pytest.approx(anim.intensity[k])
    from voxanim.next_language import SEED
    rng = np.random.default_rng(SEED)
    weight = rng.normal(size=(384, 384)).astype(np.float32)
    np.testing.assert_array_equal(anim.corner, weight[:24, :24])
    np.testing.assert_allclose(anim.rate, np.asarray(anim.batches) / anim.secs)
    lines = subtitles(anim)
    best = int(np.argmax(study.panels[0].data["ys"][0]))
    assert "peaks at batch %d" % study.panels[0].data["x"][best] in lines[1]
    assert M.number(study.panels[0].data["ys"][0][best]) in lines[1]
    assert "%.1f at batch 1" % study.panels[2].data["ys"][0][0] in lines[2]
    assert "in %d steps" % anim.done_at[0] in lines[3] and "take %d" % anim.done_at[1] in lines[3]

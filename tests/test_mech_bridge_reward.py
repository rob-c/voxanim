"""Animation 181 (voxanim.mech_bridge_reward) and its experiment
(voxanim.study_bridge_reward): the layout is clean, the loop closes, every phase says what
kind of evidence it shows, the bridge names where it comes from and where it goes, and the
numbers on screen are the study's."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_reward as M
from voxanim.mech_phys_apps import commas, pct
from voxanim.next_catalog import build_study, drawing

NAME = "labels-against-rewards"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if not _built:
        _built["a"] = (presets.get(NAME).build(), build_study("study_bridge_reward", NAME))
    return _built["a"]


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def spread(values, fmt):
    return M.span_of(values, fmt)


# ---------------------------------------------------------------------- no data needed
def test_labels_and_helpers():
    assert M.is_evidence(M.EXACT) and M.is_evidence(M.RUNS) and M.is_evidence(M.TRAIN)
    assert M.is_evidence(M.ONE_RUN) and not M.is_evidence("measured elsewhere")
    stalls = [dict(digit=2, since=100, until=400), dict(digit=4, since=100, until=1700),
              dict(digit=3, since=100, until=200)]
    assert M.stall_rows(stalls) == [(4, 100, 1700), (2, 100, 400), (3, 100, 200)]
    assert M.stall_rows(stalls, most=1) == [(4, 100, 1700)]
    assert M.stall_rows([]) == []
    assert drawing(NAME) is M.LabelsAgainstRewards


# ---------------------------------------------------------------------- the animation
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME]
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@needs_mnist
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_mnist
def test_a_bridge_from_classifiers_to_reinforcement_learning():
    a, _ = built()
    assert "from: A dense network classifies digits, trained on a label for every digit" in \
        texts(a, 0, 0.5)
    last = texts(a, len(a.PHASES) - 1, 0.95)
    assert "next: Q-learning in a gridworld" in last
    assert "Next: rewards that come late, in a gridworld" in subtitles(a)[-1]


@needs_mnist
def test_study_says_what_was_run():
    _, s = built()
    m = s.metadata
    assert len(s.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]
    assert m["evidence"]["measured"] and m["evidence"]["exact"]
    assert "moving average" in m["baseline"] and m["optimiser"] == "Adam"


# ---------------------------------------------------------------------- the numbers
@needs_mnist
def test_the_digit_and_its_gradients_on_screen_are_the_study():
    a, s = built()
    sh = s.metadata["shown"]
    y = sh["label"]
    first = texts(a, 0, 0.6)                          # a right draw
    assert "training digit %s: a %d" % (commas(sh["index"]), y) in first
    assert "the network after %d" % sh["update"] in first
    assert "%.2f" % sh["p_label"] in first
    assert M.fixed(sh["label_gradient"][y]) in first
    i = sh["draw_rewards"].index(1)
    assert "a = %d: right, r = 1; b = %.2f" % (y, sh["baseline"]) in first
    assert M.fixed(sh["draw_gradients"][i][y]) in first
    if 0 in sh["draw_rewards"]:                        # then a wrong one
        j = sh["draw_rewards"].index(0)
        wrong = texts(a, 0, 0.95)
        assert "a = %d: wrong, r = 0; b = %.2f" % (sh["draws"][j], sh["baseline"]) in wrong
        assert M.fixed(sh["draw_gradients"][j][sh["draws"][j]]) in wrong
    avg = texts(a, 1, 0.95)
    assert "mean of %s draws" % commas(sh["many"]) in avg
    assert "= %.3f × the label's gradient, for any b" % sh["p_label"] in avg
    assert "cosine with the label's: 1 if right, %.2f if wrong" % sh["wrong_cosine"] in avg
    assert "a row: (r − b)(p − onehot(a)), b = %.2f" % sh["baseline"] in avg
    assert {"a = %d, %s" % (d, "right" if d == y else "wrong") for d in sh["draws"]} <= set(avg)
    assert M.fixed(sh["mean_gradient"][y]) in avg and M.fixed(sh["expected_gradient"][y]) in avg
    assert "p(y) = %.2f times the label's" % sh["p_label"] in subtitles(a)[1]


@needs_mnist
def test_bits_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    info = m["information"]
    bits = texts(a, 2, 0.95)
    assert "a label: log₂ 10 = 3.32 bits" in bits
    assert "a right/wrong reward: at most 1 bit" in bits
    assert "seed 0, update %s: %s right, %.2f bits" % (
        commas(info["ends"][-1]), pct(info["reward_rate"][0][-1]),
        info["entropy"][0][-1]) in bits
    assert "bits in the %s digits of a run" % commas(m["examples"]) in bits
    assert "%sM" % spread(info["reward_total_bits"], lambda v: "%.2f" % (v / 1e6)) in bits
    assert "%.2fM" % (info["label_total_bits"] / 1e6) in bits
    sub = subtitles(a)[2]
    assert spread(info["peak"], lambda v: "%.2f" % v) + " bits at its peak" in sub
    assert spread(info["final_entropy"], lambda v: "%.2f" % v) in sub


@needs_mnist
def test_accuracy_on_screen_is_the_study():
    a, s = built()
    m = s.metadata
    held = texts(a, 3, 0.95)
    for level in m["levels"]:
        for kind in ("label", "reward"):
            v = m["reach"]["%.2f" % level][kind]
            if None not in v:
                assert spread(v, commas) in held
    gap = spread(m["gap_90"], lambda v: "%.1f" % v)
    assert "to 90%%: %s times the updates" % gap in held
    final = {k: spread(m["final_accuracy"][k], lambda v: pct(v, 1)) for k in ("label",
                                                                           "reward")}
    assert "rewards %s, labels %s" % (final["reward"], final["label"]) in held
    assert "labels': cosine %s on average" % spread(m["batch_cosine"]["mean"],
                                                    lambda v: "%.2f" % v) in held
    assert "%s times the updates to reach 90%%" % gap in subtitles(a)[3]


@needs_mnist
def test_the_lost_digits_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    last = texts(a, 4, 0.95)
    stalls = m["stalls"]["reward"][m["seeds"].index(m["stalled_seed"])]
    for d, since, until in M.stall_rows(stalls):
        assert "%ds: from update %s to %s" % (d, commas(since), commas(until)) in last
    assert "rewards lost a digit in %d of 3 seeds" % m["stalled_runs"] in last
    assert "labels: every digit named, every seed" in last
    assert "seed %d: each digit's held-out accuracy, updates 0 to 3,000" % m[
        "stalled_seed"] in last
    if stalls:
        d, since, until = M.stall_rows(stalls)[0]
        assert "in seed %d no %ds were named from update %s to %s" % (
            m["stalled_seed"], d, commas(since), commas(until)) in subtitles(a)[4]
    # the pictures are the study's per-digit accuracies, read at the last probe before
    np.testing.assert_allclose(a.cls_reward[:, -1], s.panels[1].data["values"][:10][
        :, int(np.flatnonzero(np.asarray(s.panels[1].data["probes"]) <= 3000)[-1])])

"""Animation 159 (voxanim.mech_hardware) and its experiment (voxanim.study_hardware):
the layout is clean over all eleven phases, the loop closes, every phase says what kind of
evidence it shows, the speeds on screen come from the one table of representative
figures (and are labelled as not measured, with no machine named), the arithmetic from
them is the stated arithmetic, the number formats are exact, and the numpy models learn
on held-out digits (and keep their accuracy in fewer bits, but for unscaled FP4)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_hardware as M
from voxanim import study_hardware as S
from voxanim.next_catalog import build_study

NAME = "cpu-and-gpu-training"
C, A = "classifier", "autoencoder"


@pytest.fixture(scope="module")
def study():
    return build_study("study_hardware", NAME)


@pytest.fixture(scope="module")
def anim():
    return presets.get(NAME).build()


def texts(a, t):
    return [x.s for x in a.scene(t).texts]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


# ---------------------------------------------------------------------- the table of speeds
def test_headline_figures_are_the_table_and_its_ratios():
    R = S.REPRESENTATIVE
    assert R["batch"] == S.BATCH == 128
    assert R["step_ms"] == {(C, "cpu"): 1.0, (C, "gpu"): 0.5, (A, "cpu"): 40.0, (A, "gpu"): 2.0}
    for key, v in R["step_ms"].items():
        assert np.isclose(S.step_ms(*key), v)         # the cost model passes through them
        assert 0 < R["fixed_ms"][key] < v
        assert np.isclose(sum(S.parts_ms(*key)), v)
    step = {k: S.step_ms(*k) for k in S.FOUR}
    assert np.isclose(step[A, "cpu"] / step[C, "cpu"], 40)    # slower on both ...
    assert np.isclose(step[A, "gpu"] / step[C, "gpu"], 4)
    assert np.isclose(step[A, "cpu"] / step[A, "gpu"], 20)    # ... yet 20x faster on the GPU
    assert np.isclose(step[C, "cpu"] / step[C, "gpu"], 2)     # and the small one only 2x
    # the fixed cost is most of the small model's GPU step, and little of the large one's
    assert R["fixed_ms"][C, "gpu"] / step[C, "gpu"] >= 0.8
    assert R["fixed_ms"][A, "gpu"] / step[A, "gpu"] <= 0.4
    assert [round(S.epoch_s(*k), 3) for k in S.FOUR] == [0.468, 0.234, 18.72, 0.936]
    assert S.parts_ms(C, "cpu")[1] == 0 and S.parts_ms(C, "gpu")[:2] == (0.05, 0.1)
    assert R["copy_ms"] == {C: 0.5, A: 5.0}


def test_cost_model_is_a_straight_line_and_the_gpu_pulls_ahead():
    for key in S.FOUR:
        t = [S.step_ms(*key, b) for b in S.BATCHES]
        assert np.allclose(np.diff(t) / np.diff(S.BATCHES), S.per_digit_ms(*key))
        assert np.isclose(S.step_ms(*key, 0), S.REPRESENTATIVE["fixed_ms"][key])
    for name in S.NAMES:                              # GPU: larger fixed cost, smaller slope
        assert S.REPRESENTATIVE["fixed_ms"][name, "gpu"] > S.REPRESENTATIVE["fixed_ms"][name,
                                                                                      "cpu"]
        assert S.per_digit_ms(name, "gpu") < S.per_digit_ms(name, "cpu") / 10
        gain = [S.step_ms(name, "cpu", b) / S.step_ms(name, "gpu", b) for b in S.BATCHES]
        assert (np.diff(gain) > 0).all()
    assert S.crossover(C) == 64 and S.step_ms(C, "gpu", 32) > S.step_ms(C, "cpu", 32)
    assert S.crossover(A) == S.BATCHES[0]             # the large model: ahead at every size


# ---------------------------------------------------------------------- the models
def test_sizes_and_arithmetic_of_the_two_models():
    assert S.parameters(C) == 784 * 128 + 128 + 128 * 10 + 10 == 101770
    assert S.parameters(A) == 5379920
    assert S.multiply_adds(C, 1) == 2 * (784 * 128 + 128 * 10) + 128 * 10
    assert S.multiply_adds(A, 128) == 128 * S.multiply_adds(A, 1)
    assert S.parameters(A) > 50 * S.parameters(C)
    P = S.init_weights(A, 0)
    assert all(v.dtype == np.float32 for v in P.values())
    assert sum(v.size for v in P.values()) == S.parameters(A)
    assert np.array_equal(S.init_weights(C, 3)["W0"], S.init_weights(C, 3)["W0"])
    assert not np.array_equal(S.init_weights(C, 3)["W0"], S.init_weights(C, 4)["W0"])


def test_batch_order_covers_each_epoch_once():
    order = S.batch_order(5, 2, batch=100, n=1000)
    assert order.shape == (20, 100)
    for epoch in (order[:10], order[10:]):
        assert sorted(epoch.ravel().tolist()) == list(range(1000))
    assert not np.array_equal(order[:10], order[10:])
    assert S.batch_order(0, 1).shape == (S.N_TRAIN // S.BATCH, S.BATCH)


@pytest.mark.parametrize("name", S.NAMES)
def test_gradients_match_finite_differences(name, monkeypatch):
    sizes = (6, 5, 3) if name == C else (6, 4, 6)
    monkeypatch.setitem(S.MODELS, name, dict(S.MODELS[name], sizes=sizes))
    rng = np.random.default_rng(1)
    P = {k: v.astype(np.float64) + 0.1 * rng.normal(size=v.shape)
         for k, v in S.init_weights(name, 0).items()}
    x, y = rng.random((7, 6)), rng.integers(0, 3, 7)
    _, grads = S.loss_and_grads(name, P, x, y)
    for key in P:
        i = tuple(int(v) // 2 for v in P[key].shape)
        up, down = ({k: v.copy() for k, v in P.items()} for _ in range(2))
        up[key][i] += 1e-5
        down[key][i] -= 1e-5
        slope = (S.loss_and_grads(name, up, x, y)[0]
                 - S.loss_and_grads(name, down, x, y)[0]) / 2e-5
        assert np.isclose(grads[key][i], slope, rtol=2e-3, atol=1e-6), key


def test_a_step_stays_in_float32_and_lowers_the_loss():
    from voxanim.optim import Adam

    rng = np.random.default_rng(0)
    x, y = rng.random((32, 784)).astype(np.float32), rng.integers(0, 10, 32)
    P = S.init_weights(C, 0)
    opt = Adam(P, S.FLOAT(S.RATE))
    losses = [S.numpy_step(C, P, opt, x, y) for _ in range(30)]
    assert all(v.dtype == np.float32 for v in P.values())
    assert losses[-1] < 0.5 * losses[0]


def test_durations_read_as_a_person_would_say_them():
    assert S.duration(0.00053) == "530 µs" and S.duration(0.468) == "468 ms"
    assert S.duration(3.36) == "3.4 s" and S.duration(93.6) == "94 s"
    assert S.duration(374.4) == "6.2 min" and S.duration(1872000) == "22 days"
    assert S.duration(4e8) == "12.7 years"
    assert M.ms(0.05) == "0.05 ms" and M.ms(1.0) == "1 ms" and M.ms(40.0) == "40 ms"
    assert M.times(2.0) == "2×" and M.times(2.64) == "2.6×" and M.times(52.9) == "53×"
    assert M.count(101770) == "101,770" and M.count(5379920) == "5.4 million"
    assert M.short(101770) == "101,770" and M.short(5379920) == "5.4M"
    assert M.CpuAndGpuTraining.versus(20.0) == "20× faster"
    assert M.CpuAndGpuTraining.versus(0.374) == "2.7× slower"


# ---------------------------------------------------------------------- the animation
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=60, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_long_and_loop_closes(anim):
    assert type(anim) is M.ANIMATIONS[NAME]
    assert 8 <= len(anim.PHASES) <= 11
    assert 150.0 <= anim.t_done <= 200.0 and anim.duration <= 205.0   # long: about waiting

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for lo, hi in anim.spans:                         # every phase draws things, not only text
        assert len(anim.scene((lo + hi) / 2).quads) >= 20


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds(anim):
    for t in np.arange(0.0, anim.duration, 0.37):     # no instant between the stills crashes
        anim.scene(float(t))


@needs_mnist
def test_the_map_stays_on_screen_through_every_phase(anim):
    parts = {"RAM (main memory)", "storage", "CPU", "GPU", "GPU memory", "bus"}
    for k in range(len(anim.PHASES)):
        for u in (0.1, 0.5, 0.97):
            assert parts <= set(texts(anim, at(anim, k, u)))


@needs_mnist
def test_weights_blocks_are_readable_and_say_how_many(anim):
    assert anim.cells == {C: 6, A: 16}                # a compressed scale: both are grids
    assert anim.side[C] >= 20 and anim.side[A] > 2 * anim.side[C]
    assert "101,770 weights" in texts(anim, at(anim, 1, 0.5))
    assert "5.4M weights" in texts(anim, at(anim, 6, 0.5))
    assert "the weights: 101,770 numbers" in texts(anim, at(anim, 2, 0.5))


@needs_mnist
def test_subtitles_fit_two_lines_and_evidence_is_labelled(anim):
    for k in range(len(anim.PHASES)):
        lines = layout.wrap(anim.subtitle_for(k), layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2, k
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    assert anim.FIGURES == "representative figures, not measured"
    assert anim.ARITHMETIC == "arithmetic from representative figures"
    assert anim.FIGURES in anim.RACED and "measured on MNIST" in anim.RACED
    assert anim.FIGURES in anim.BITS and "measured on MNIST" in anim.BITS
    kinds = (anim.PICTURE, anim.FIGURES, anim.RACED, anim.ARITHMETIC, anim.BITS)
    shown = []
    for k in range(len(anim.PHASES)):                 # every phase says what kind of evidence
        for u in (0.1, 0.5, 0.9):
            got = [t for t in texts(anim, at(anim, k, u)) if t in kinds]
            assert len(got) == 1, (k, u)
        shown.append(got[0])
    # every phase that shows a speed says the speeds are not measured
    assert shown == ([anim.PICTURE] + [anim.FIGURES] * 3 + [anim.BITS] + [anim.FIGURES] * 3
                     + [anim.RACED] + [anim.ARITHMETIC] * 2)
    subs = " ".join(anim.subtitle_for(k) for k in range(len(anim.PHASES)))
    assert "typical round figures" in anim.subtitle_for(1)       # said once, early
    assert "real machines differ" in anim.subtitle_for(1)
    assert "By the same arithmetic" in subs


@needs_mnist
def test_no_machine_is_named_and_nothing_claims_a_measured_speed(anim, study):
    import inspect

    words = ("Apple", "M5", "macOS", "Metal", "PyTorch", "torch", "Intel", "Nvidia", "NVIDIA",
             "128 GB", "18 cores")
    seen = set()
    for t in np.arange(0.0, anim.duration, 0.5):
        seen |= set(texts(anim, float(t)))
    seen |= {anim.subtitle_for(k) for k in range(len(anim.PHASES))}
    for text in seen:
        assert not any(w in text for w in words), text
        if "measured" in text:                        # only "not measured", or the curves
            assert "not measured" in text or "measured on MNIST" in text, text
    blob = repr(study.metadata) + inspect.getsource(S) + inspect.getsource(M)
    assert not any(w in blob for w in ("Apple", "M5 Max", "macOS", "hardware_timings"))
    m = study.metadata
    assert m["speeds_measured"] is False
    assert "not measured" in m["source"] and "owner" in m["evidence"]["speeds"]
    assert any("were not measured" in s for s in m["limitations"])


@needs_mnist
def test_numbers_on_screen_come_from_the_table(anim, study):
    m, R = study.metadata, S.REPRESENTATIVE
    step = {k: m["step_ms"]["%s/%s" % k] for k in S.FOUR}
    assert all(np.isclose(step[k], R["step_ms"][k]) for k in S.FOUR)
    assert anim.step == step
    # the four bars, with their epochs, and the ratios
    seen = texts(anim, at(anim, 7, 0.9))
    for k in S.FOUR:
        assert "%s; an epoch %s" % (M.ms(R["step_ms"][k]), S.duration(S.epoch_s(*k))) in seen
    assert "1 ms; an epoch 468 ms" in seen and "40 ms; an epoch 19 s" in seen
    assert "autoencoder: the GPU is 20× faster" in seen
    assert "classifier: the GPU is 2× faster" in seen
    sub = anim.subtitle_for(7)
    assert "40× the classifier's time on the CPU and 4× on the GPU" in sub
    assert "still 20× faster" in sub and "only 2× faster" in sub
    assert "about 1 ms a step on the CPU, 0.5 ms on the GPU, only 2× faster" in \
        anim.subtitle_for(5)
    assert "About 40 ms a step on the CPU and 2 ms on the GPU" in anim.subtitle_for(6)
    assert "the GPU is 20× faster" in anim.subtitle_for(6)
    # the parts of a GPU step, and the copy of the weights
    seen = texts(anim, at(anim, 2, 0.5))
    for label in (M.ms(R["copy_ms"][C]), M.ms(R["pick_ms"]), M.ms(R["move_ms"]),
                  M.ms(R["step_ms"][C, "gpu"] - R["pick_ms"] - R["move_ms"])):
        assert label in seen
    assert "arithmetic 0.05 ms (10%); the rest is fixed" in texts(anim, at(anim, 3, 0.9))
    assert "arithmetic 1.4 ms (70%); the rest is fixed" in texts(anim, at(anim, 3, 0.9))
    # the models' sizes
    seen = texts(anim, at(anim, 6, 0.6))
    assert "784 → 2048 → 512 → 64 → 512 → 2048 → 784" in seen
    assert any(t.startswith("5.4 million weights (53×)") for t in seen)
    assert "40 ms a step" in seen and "2 ms a step" in seen
    assert any("cost model" in t or "(a model)" in t for t in seen)
    # the step counter reaches a whole epoch, on the table's clock
    seen = texts(anim, at(anim, 1, 0.985))
    assert "steps done: %d" % (S.N_TRAIN // S.BATCH) in seen
    assert S.duration(S.epoch_s(C, "cpu")) in seen
    # the loss on screen is the real first loss of the numpy run
    first = float(S.curves()["classifier_first"][0, 0])
    assert "loss of this batch: %.2f" % first in texts(anim, at(anim, 1, 0.4))


@needs_mnist
def test_the_race_runs_on_a_scaled_clock_that_says_so(anim, study):
    m = study.metadata
    for k in S.FOUR:
        assert np.isclose(anim.race_s[k], m["epochs"] * S.epoch_s(*k))
    assert anim.race_span >= max(anim.race_s.values())     # the slowest finishes on screen
    assert anim.race_scale == int(anim.race_scale) > 1     # 94 s does not fit at 1:1
    lo, hi = anim.RACE
    done = texts(anim, at(anim, 8, hi + 0.05))
    assert "clock runs %d× fast" % anim.race_scale in done
    assert "running %d× fast" % anim.race_scale in anim.subtitle_for(8)
    for k in S.FOUR:
        assert any(t.endswith("in " + S.duration(anim.race_s[k])) for t in done)
    part = texts(anim, at(anim, 8, lo + (hi - lo) * 0.5 * min(anim.race_s.values())
                          / anim.race_span))
    assert not any(" in " in t and ("loss" in t or "right" in t) for t in part)   # none done
    # the contrast: when the autoencoder's GPU lane is done its CPU lane has hardly begun
    mid = texts(anim, at(anim, 8, lo + (hi - lo) * 1.2 * anim.race_s[A, "gpu"]
                         / anim.race_span))
    assert sum(" in " in t and ("loss" in t or "right" in t) for t in mid) == 3
    assert anim.race_s[A, "gpu"] < 0.06 * anim.race_s[A, "cpu"]
    assert "%.1f%% right in %s" % (100 * np.mean(m["final_accuracy"]),
                                   S.duration(anim.race_s[C, "cpu"])) in done


@needs_mnist
def test_projections_are_the_stated_arithmetic(anim, study):
    m = study.metadata
    assert [(p["model"], p["data"]) for p in m["projections"]] == [(1, 1), (100, 1), (10000, 1),
                                                                   (100, 1000)]
    for p in m["projections"]:
        for dev in S.DEVICES:
            want = m["projection_epochs"] * S.epoch_s(A, dev) * p["model"] * p["data"]
            assert np.isclose(p["seconds"][dev], want)
    first, last = m["projections"][0]["seconds"], m["projections"][-1]["seconds"]
    assert S.duration(first["cpu"]) == "6.2 min" and S.duration(first["gpu"]) == "19 s"
    assert S.duration(last["cpu"]) == "433 days" and S.duration(last["gpu"]) == "22 days"
    assert "22 days on the GPU, 433 days on the CPU" in anim.subtitle_for(9)
    seen = texts(anim, at(anim, 9, 0.95))
    assert "GPU 22 days, CPU 433 days" in seen
    assert "a video app on a phone opens:" in seen


# ---------------------------------------------------------------------- the study
@needs_mnist
def test_study_says_what_was_run(study):
    m = study.metadata
    assert len(study.panels) == 4 and m["seeds"] == [0, 1, 2] and m["precision"] == "float32"
    assert m["designs_tried"] and m["limitations"]
    assert m["representative"]["step_ms"]["autoencoder/gpu"] == 2.0
    assert "cost per digit" in m["evidence"]["cost_model"]
    assert m["split_sizes"] == [60000, 10000]
    assert m["models"][A]["multiply_adds_per_step"] == S.multiply_adds(A, S.BATCH)
    assert np.allclose(study.panels[1].data["ys"][:, S.BATCHES.index(S.BATCH)],
                       study.panels[0].data["values"])


@needs_mnist
def test_both_models_learn_on_held_out_digits_from_every_seed(study):
    acc = np.asarray(study.panels[2].data["ys"])
    ae = np.asarray(study.panels[3].data["ys"])
    assert acc.shape[0] == ae.shape[0] == 3
    assert (acc[:, 0] < 0.25).all() and (acc[:, -1] > 0.96).all()
    assert (ae[:, -1] < 0.1 * ae[:, 0]).all()
    assert (ae[:, -1] < ae[:, 6] - 0.001).all()        # still falling after the first epoch
    steps = np.asarray(study.panels[2].data["x"])
    assert steps[0] == 0 and steps[-1] == S.EPOCHS * (S.N_TRAIN // S.BATCH)


@needs_mnist
def test_pictures_are_the_real_digits_and_the_real_first_batch():
    c = S.curves()
    x, y, xt, yt = S.mnist_data()
    rows = S.batch_order(S.SEEDS[0], 1)[0]
    assert np.array_equal(c["batch_rows"], rows)
    assert np.array_equal(c["batch_digits"], x[rows[:16]])
    assert np.array_equal(c["batch_labels"], y[rows[:16]])
    assert np.array_equal(c["held_digits"], xt[:S.SHOWN])
    # the loss of the first step, recomputed
    loss, _ = S.loss_and_grads(C, S.init_weights(C, S.SEEDS[0]), x[rows], y[rows])
    assert np.isclose(loss, c["classifier_first"][0, 0])
    # the square filled in phase 4 is that batch through the first layer
    P = S.init_weights(C, S.SEEDS[0])
    assert np.allclose(c["hidden"], np.maximum(x[rows] @ P["W0"] + P["b0"], 0))
    # the trained autoencoder gives back something close to the digit; the classifier names it
    assert c["autoencoder_shown"].shape == (S.SHOWN, 784)
    assert ((c["autoencoder_shown"] - xt[:S.SHOWN]) ** 2).mean() < 0.02
    assert (c["classifier_shown"].argmax(1) == yt[:S.SHOWN]).mean() >= 0.75


# ---------------------------------------------------------------------- precision
def test_formats_are_exact():
    assert [S.BITS[n] for n in S.FORMAT_NAMES] == [64, 32, 16, 16, 8, 4]
    assert S.stored("FP64") == (0.1, float(abs(Fraction(0.1) - Fraction(1, 10))))
    assert S.stored("FP32")[0] == float(np.float32(0.1))
    assert S.stored("FP16")[0] == float(np.float16(0.1)) == 0.0999755859375
    assert S.stored("BF16")[0] == 0.10009765625                 # the top 16 bits of float32
    assert S.stored("FP8")[0] == 0.1015625 and S.stored("FP4") == (0.0, 0.1)
    assert list(S.small_grid("FP4")) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    g = S.small_grid("FP8")
    assert g[-1] == 448 and g[1] == 2.0 ** -9 and len(g) == 127
    # every value of a grid is kept exactly; beyond the largest saturates; ties go even
    assert np.array_equal(S.to_format("FP8", g), g) and S.to_format("FP4", [7.0, -9.0]).tolist() \
        == [6.0, -6.0]
    assert S.to_format("FP4", [1.25, 1.75, 2.5]).tolist() == [1.0, 2.0, 2.0]
    errs = [S.stored(n)[1] for n in ("FP64", "FP32", "FP16", "FP8", "FP4")]
    assert errs == sorted(errs)                       # fewer fraction bits, a larger error


def test_block_scale_keeps_small_weights():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, (64, 3))
    plain, scaled = S.to_format("FP4", W), S.to_format_scaled("FP4", W)
    assert (plain == 0).mean() > 0.9 and (scaled == 0).mean() < 0.2
    assert np.abs(scaled - W).mean() < 0.2 * np.abs(W).mean()
    for s in (0, 32):                                 # one power-of-two scale per block
        block = scaled[s:s + 32, 0]
        nz = block[block != 0]
        ratios = np.abs(nz) / np.abs(nz).min()
        assert np.allclose(ratios * 2, np.round(ratios * 2))


def test_representative_speeds_at_each_width():
    m = S.REPRESENTATIVE["madds_per_s"]
    assert m["cpu", "FP32"] == 2 * m["cpu", "FP64"]              # the CPU: 64-bit natively
    assert m["gpu", "FP32"] / m["gpu", "FP64"] == 64              # the GPU: built for 32-bit
    assert m["gpu", "FP64"] < m["cpu", "FP64"]                    # ... and slower than the CPU
    assert [m["gpu", f] / m["gpu", "FP32"] for f in ("FP16", "FP8", "FP4")] == [4, 8, 16]
    # the 20x between the two at 32-bit is the autoencoder step's 20x
    assert m["gpu", "FP32"] / m["cpu", "FP32"] == S.step_ms(A, "cpu") / S.step_ms(A, "gpu")


@needs_mnist
def test_fewer_bits_keep_the_accuracy_but_for_unscaled_fp4(study):
    got = study.metadata["weights_in_fewer_bits"]
    assert set(got) == set(S.VARIANTS)
    full = got["FP32"]["accuracy"]
    assert full > 0.96 and abs(got["FP64"]["accuracy"] - full) < 0.002
    for v in ("BF16", "FP16", "FP8", "FP8 scaled", "FP4 scaled"):
        assert abs(got[v]["accuracy"] - full) < 0.01, v
    assert got["FP4"]["accuracy"] < 0.5 and got["FP4"]["zero_weights"] > 0.9
    assert got["FP4 scaled"]["zero_weights"] < 0.2
    f = study.metadata["formats"]
    assert np.isclose(f["FP32"]["autoencoder_mb"], S.REPRESENTATIVE["megabytes"][A], rtol=0.01)
    assert np.isclose(f["FP64"]["autoencoder_mb"], 16 * f["FP4"]["autoencoder_mb"])


@needs_mnist
def test_precision_phase_shows_the_numbers(anim, study):
    k = [name for name, _ in anim.PHASES].index("precision")
    assert k == 4 and anim.PHASES[k - 1][0] == "cores"
    sub = anim.subtitle_for(k)
    assert "64-bit" in sub and "4-16× faster" in sub and "64× slower" in sub
    seen = set()
    for u in np.linspace(0.05, 0.98, 24):
        seen |= set(texts(anim, at(anim, k, u)))
    assert {"64-bit", "32-bit; also 16, 8, 4", anim.BITS} <= seen
    for name in S.FORMAT_NAMES:
        assert name in seen
    assert {"0.1000000015", "0.1000976562", "0.09997558594", "0.1015625"} <= seen
    assert "FP4 holds only 8 sizes: 0, 0.5, 1, 1.5, 2, 3, 4, 6" in seen
    assert {"0.25", "0.5", "0.156", "10", "40", "80", "160"} <= seen
    assert "GPU: 64-bit at 1/64 of its 32-bit: below the CPU's" in seen
    got = study.metadata["weights_in_fewer_bits"]
    assert "%.1f%%" % (100 * got["FP4"]["accuracy"]) in seen
    assert "%.1f%%" % (100 * got["FP4 scaled"]["accuracy"]) in seen
    assert "plain FP4 rounds %d%% of the weights to 0;" % round(100 * got["FP4"]["zero_weights"]) \
        in seen

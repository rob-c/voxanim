"""Animations 86-90 (voxanim.mech_systems) and the redesigned experiments they
draw (voxanim.study_systems): the layout holds, every number on screen is the
study's, and each study supports the claim its animation makes."""

import itertools

import numpy as np
import pytest
from conftest import available, needs_mnist

from voxanim import mech_systems as M
from voxanim import qa
from voxanim import study_systems as S
from voxanim.next_catalog import build, build_study

NAMES = ("data-parallel-allreduce", "tensor-and-pipeline-parallel", "zero-sharding",
         "activation-checkpointing", "roofline")
# the catalogue lists it as needing no data; its redesigned study trains on MNIST
USES_MNIST = ("activation-checkpointing",)


def needs(name):
    if name in USES_MNIST:
        return needs_mnist
    return pytest.mark.skipif(not available(name), reason="%s needs data not downloaded" % name)


def study(name):
    return build_study("next_systems", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def fd_gradient(params, x, y, eps=1e-6):
    """The gradient of the mean cross-entropy by central differences."""
    out = []
    for layer in params:
        pair = []
        for t in layer:
            g = np.zeros_like(t)
            for idx in np.ndindex(t.shape):
                old = t[idx]
                t[idx] = old + eps
                up = S.xent(S.mlp_forward(params, x)[-1], y)
                t[idx] = old - eps
                down = S.xent(S.mlp_forward(params, x)[-1], y)
                t[idx] = old
                g[idx] = (up - down) / (2 * eps)
            pair.append(g)
        out.append(pair)
    return out


def small_mlp(sizes, seed=3, n=5):
    rng = np.random.default_rng(seed)
    params = S.mlp_init(sizes, rng)
    for w, b in params:  # noqa: B007
        b += rng.normal(0, 0.1, b.shape)
    return params, rng.normal(size=(n, sizes[0])), rng.integers(0, sizes[-1], n)


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in NAMES])
def test_layout_and_loop(name):
    assert qa.check(name, samples=24, verbose=False) == 0
    anim = build("next_systems", name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.duration <= 60.0
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    # the loop closes on an empty frame: nothing but the title block at either end
    assert sorted(t.s for t in first.texts) == sorted(t.s for t in last.texts)
    assert len(first.lines) == len(last.lines) == 0
    assert all(subtitles(anim))


def test_the_drawings_compute_nothing_for_themselves():
    """The work-arounds are gone: no arithmetic of an experiment lives in the drawing module."""
    for name in ("middle_pixels", "ring_allreduce", "split_layer", "pipeline_timeline", "chain",
                 "traced_backprop", "OPERATIONS"):
        assert not hasattr(M, name)
    assert set(S.STUDIES) == set(NAMES) - {"zero-sharding"}


# ---------------------------------------------------------------------- 86
def test_ring_allreduce_is_exact_on_integers():
    rng = np.random.default_rng(0)
    v = rng.integers(-50, 50, (4, 22)).astype(float) * 4        # uneven chunks: 6, 6, 5, 5
    ring = S.ring_allreduce(v)
    np.testing.assert_array_equal(ring["gather"][-1], np.broadcast_to(v.mean(0), v.shape))
    assert list(np.diff(ring["edges"])) == [6, 6, 5, 5]
    assert (ring["counts"][0] == 1).all()
    for r in range(4):
        c = (r + 1) % 4
        assert ring["counts"][-1][r, c] == 4                     # rank r ends owning chunk r + 1
        part = slice(ring["edges"][c], ring["edges"][c + 1])
        np.testing.assert_array_equal(ring["reduce"][-1][r, part], v[:, part].sum(0))
    assert ring["have"][0].sum() == 4 and ring["have"][-1].all()
    # every pass moves one chunk per worker, to its right-hand neighbour
    for sends in list(ring["reduce_sends"]) + list(ring["gather_sends"]):
        assert sorted(int(w) for w, _, _ in sends) == [0, 1, 2, 3]
        assert all(to == (w + 1) % 4 for w, _, to in sends)
    # a worker sends every chunk but one, twice: 2 (P - 1) / P of the vector when chunks are even
    even = S.ring_allreduce(rng.normal(size=(4, 24)))
    assert (even["sent"] == 2 * 3 * 6 * 8).all()


def test_mlp_gradient_matches_finite_differences():
    params, x, y = small_mlp((4, 3, 3, 2))
    for got, want in zip(S.mlp_grad(params, x, y), fd_gradient(params, x, y)):
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, atol=1e-7)


@needs("data-parallel-allreduce")
def test_allreduce_study_averaged_gradient_is_the_big_batch_gradient():
    s = study("data-parallel-allreduce")
    first, reduce, gather, training = (p.data for p in s.panels)
    local, whole = first["local"], gather["whole"]
    assert local.shape == (4, 12, 8)
    # data integrity: 64 real digits, 16 a worker, and a held-out split from the test set
    assert first["digits"].shape == (64, 784) and 0 <= first["digits"].min() < 1
    assert first["digits"].max() == 1 and (first["digits"] > 0.5).mean(1).min() > 0.02
    assert set(np.unique(first["labels"])) <= set(range(10)) and len(set(first["labels"])) > 5
    data = s.metadata["data"]
    assert data["train"]["split"] == "train" and data["heldout"]["split"] == "test"
    assert data["train"]["x_sha256"] != data["heldout"]["x_sha256"]
    # the workers' gradients really differ, and no chunk of them is blank
    off = ~np.eye(4, dtype=bool)
    assert first["cosines"][off].max() < 0.8
    assert not np.allclose(local[0], local[1], atol=1e-3)
    assert np.abs(local).reshape(4, 4, -1).max(-1).min() > 1e-3
    # ... yet their average, as the ring delivers it to every worker, is the whole-batch gradient
    np.testing.assert_allclose(local.mean(0), whole, atol=1e-12)
    np.testing.assert_allclose(gather["values"], np.broadcast_to(whole.ravel(), (4, 96)),
                               atol=1e-12)
    assert (reduce["counts"][0] == 1).all() and gather["have"][-1].all()
    for r in range(4):
        c = (r + 1) % 4
        np.testing.assert_allclose(reduce["states"][-1][r, 24 * c:24 * c + 24],
                                   local.reshape(4, 4, 24)[:, c].sum(0), atol=1e-12)
    assert training["average_error"].max() < 1e-12              # at every one of the 200 steps
    # one step with either gradient gives the same weights, and so does the whole run
    assert training["step_apart"] < 1e-12
    assert np.ptp(training["step_loss"][1:]) < 1e-12
    assert training["weights_apart"].max() < 1e-9
    ring, one = training["ys"]
    np.testing.assert_allclose(ring, one, atol=1e-9)
    assert ring[-1] < ring[0] - 1.0                              # and it is real training
    assert training["accuracy"][0, -1] > 0.6 > 0.2 > training["accuracy"][0, 0]


@needs("data-parallel-allreduce")
def test_allreduce_numbers_are_the_studys():
    s = study("data-parallel-allreduce")
    a = M.DataParallelAllReduce(s)
    size, ring = (int(v) for v in s.panels[2].data["bytes"])
    assert (a.grad_bytes, a.ring_bytes) == (size, ring) == (768, 1152)
    assert f"{ring:,}" in a.subtitle_for(3) and str(size) in a.subtitle_for(3)
    assert 6 * a.per_pass == ring                                # six passes of one chunk each
    # what is drawn is the study's own arrays
    np.testing.assert_array_equal(a.d["local"], s.panels[0].data["local"])
    np.testing.assert_array_equal(a.d["reduce"].reshape(4, 4, 96), s.panels[1].data["states"])
    np.testing.assert_array_equal(a.d["gather"].reshape(4, 4, 96), s.panels[2].data["states"])
    np.testing.assert_array_equal(a.d["full"], s.panels[2].data["whole"])
    assert a.err < 1e-12 and M.number(a.err) in a.subtitle_for(3)
    loss = s.panels[3].data["ys"]
    assert "%.2f to %.2f" % (loss[0, 0], loss[0, -1]) in a.subtitle_for(4)
    assert "%d training steps in" % s.metadata["shown_step"] in a.subtitle_for(0)


# ---------------------------------------------------------------------- 87
def test_pipeline_runs_every_micro_batch_through_every_stage():
    params, x, _ = small_mlp((4, 3, 3, 2), n=6)
    out, timeline = S.run_pipeline(params, x, 3)
    np.testing.assert_allclose(out, S.mlp_forward(params, x)[-1], atol=1e-14)
    np.testing.assert_array_equal(timeline, [[1, 2, 3, 0, 0], [0, 1, 2, 3, 0], [0, 0, 1, 2, 3]])


@needs("tensor-and-pipeline-parallel")
def test_split_layer_gives_the_same_output():
    s = study("tensor-and-pipeline-parallel")
    columns, rows = s.panels[0].data, s.panels[1].data
    x, w, reference = columns["x"], columns["w"], columns["reference"]
    assert x.shape == (16, 12) and w.shape == (12, 16)
    # a trained layer on real digits: no blank inputs, nothing left of the blank corner
    assert (np.abs(x) > 0.05).mean() > 0.9 and (np.abs(x) > 0.05).any(0).all()
    np.testing.assert_allclose(reference, x @ w, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(columns["cols"], 1), reference, atol=1e-12)
    np.testing.assert_allclose(rows["parts"].sum(0), reference, atol=1e-12)
    assert not np.allclose(rows["parts"][0], reference, atol=0.1)
    # the classifier is a trained one, and splitting its layer changes nothing
    whole, by_column, by_row = rows["accuracy"]
    assert whole > 0.8 and whole == by_column == by_row
    assert rows["logit_gap"].max() < 1e-10
    data = s.metadata["data"]
    assert data["train"]["split"] == "train" and data["heldout"]["split"] == "test"
    a = M.TensorAndPipelineParallel(s)
    np.testing.assert_array_equal(a.d["x"], x)
    np.testing.assert_array_equal(a.d["parts"], rows["parts"])
    assert a.err_rows < 1e-12 and a.err_cols < 1e-12
    assert M.number(a.err_rows) in a.subtitle_for(1)


@needs("tensor-and-pipeline-parallel")
def test_pipeline_numbers_are_the_studys():
    s = study("tensor-and-pipeline-parallel")
    a = M.TensorAndPipelineParallel(s)
    timeline = np.asarray(s.panels[2].data["values"]).astype(int)
    assert timeline.shape == (4, 11) and s.panels[2].data["gap"] < 1e-10
    for stage in range(4):                                       # one step later per stage
        np.testing.assert_array_equal(timeline[stage, stage:stage + 8], np.arange(1, 9))
    assert "%d of %d slots" % ((timeline > 0).sum(), timeline.size) in a.subtitle_for(2)
    bubble = s.panels[3].data
    micro, util = bubble["x"], np.asarray(bubble["ys"])[0]
    assert list(micro) == [1, 2, 4, 8, 16, 32] and bubble["gap"].max() < 1e-10
    for m, u, t in zip(micro, util, a.sweep):
        assert t.shape == (4, m + 3) and (t >= 0).all()
        assert (t > 0).mean() == pytest.approx(u) == pytest.approx(m / (m + 3))
        assert (t == 0).sum() == 12                              # the bubble never changes size
    assert "%.0f%% with 1 and %.0f%% with 32" % (100 * util[0], 100 * util[-1]) in a.subtitle_for(3)


# ---------------------------------------------------------------------- 88
def test_zero_numbers_are_the_studys():
    s = study("zero-sharding")
    a = M.ZeroSharding(s)
    per = np.asarray(s.panels[2].data["values"], float)
    np.testing.assert_allclose(per, [112, 38.5, 26.25, 14])
    assert a.devices == 8 and a.params == 7
    # the bytes per parameter the first phase spells out are the study's
    np.testing.assert_allclose(7.0 * np.array(a.BYTES), s.panels[0].data["values"])
    np.testing.assert_allclose(a.parts.sum(1), per)
    assert "112 GB" in a.subtitle_for(0)
    assert "38.5 GB instead of 112" in a.subtitle_for(1)
    assert "26.25 GB" in a.subtitle_for(2)
    assert "14 GB per device, an 8th of 112" in a.subtitle_for(3)


def test_zero_stage_heights_end_on_the_studys():
    a = M.ZeroSharding(study("zero-sharding"))
    for stage in (1, 2, 3):
        np.testing.assert_allclose(a._held(stage, 0.0), a.parts[stage - 1])
        np.testing.assert_allclose(a._held(stage, 1.0), a.parts[stage])
    flat, sharded = a.scaling
    np.testing.assert_allclose(sharded, 112.0 / a.counts)
    assert (flat == 112).all()


# ---------------------------------------------------------------------- 89
def test_checkpointed_gradient_matches_finite_differences():
    params, x, y = small_mlp((4, 3, 3, 3, 3, 2))
    want = fd_gradient(params, x, y)
    for spacing in (1, 2, 3, 5):
        got, stats = S.checkpointed_grad(params, x, y, spacing)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g[0], w[0], atol=1e-7)
            np.testing.assert_allclose(g[1], w[1], atol=1e-7)
        assert stats["calls"] >= 5
    # spacing 1 is plain backpropagation: every activation held, every layer run once
    assert S.checkpointed_grad(params, x, y, 1)[1]["peak"] == 6
    assert S.checkpointed_grad(params, x, y, 1)[1]["calls"] == 5


@needs("activation-checkpointing")
def test_checkpointing_study_trades_memory_for_compute():
    s = study("activation-checkpointing")
    mem, work, grad, same = (p.data for p in s.panels)
    ks = [int(k) for k in mem["spacings"]]
    peak, runs = dict(zip(ks, mem["values"])), dict(zip(ks, work["values"]))
    n = 16
    assert ks == list(range(1, n + 1))
    assert (peak[1], runs[1]) == (n + 1, n)                      # plain backpropagation
    for k in (1, 2, 4, 8, 16):                                   # the exact counts
        assert peak[k] == n // k + k and runs[k] == 2 * n - n // k
    # the trade: less memory costs more layer runs, and the square root is a minimum of memory
    assert peak[4] == min(peak.values()) == 8 < peak[1] - 8
    assert peak[2] > peak[4] < peak[8] < peak[16]
    assert all(runs[a] <= runs[b] for a, b in itertools.pairwise(ks)) and runs[16] == 2 * n - 1
    assert runs[4] > runs[1] + 8
    assert mem["bytes"][3] == mem["bytes"].min() < 0.6 * mem["bytes"][0]
    # the gradient does not change, at any spacing
    assert same["values"].max() <= 1e-12 * same["scale"]
    np.testing.assert_allclose(grad["checkpointed"][3], grad["plain"], atol=1e-14)
    assert np.abs(grad["plain"]).max() > 1e-3
    # a real, trained, deep classifier on held-out digits
    assert s.metadata["heldout_loss_accuracy_after"][1] > 0.8
    assert s.metadata["heldout_loss_accuracy_before"][1] < 0.2
    data = s.metadata["data"]
    assert data["train"]["split"] == "train" and data["heldout"]["split"] == "test"
    assert grad["digits"].shape == (8, 784) and grad["digits"].max() == 1


@needs("activation-checkpointing")
def test_checkpointing_numbers_are_the_studys():
    s = study("activation-checkpointing")
    a = M.ActivationCheckpointing(s)
    mem, work = s.panels[0].data["values"], s.panels[1].data["values"]
    assert a.spacings == list(range(1, 17)) and a.k == 4 and a.least == [3, 4, 5, 6]
    assert a.mem == [int(v) for v in mem] and a.work == [int(v) for v in work]
    for k, m, w in zip(a.spacings, mem, work):
        events = a.runs[k]
        assert max(M.held(e) for e in events) == m               # the peak the study measured
        assert events[-1]["calls"] == w
        assert events[-1]["done"] == set(range(16)) and events[-1]["temp"] == set()
        # only checkpoints and the logits are ever stored
        assert all(i % k == 0 or i == 16 for e in events for i in e["kept"])
        # one forward pass, then the last segment: recomputed unless it is a single layer
        last = 16 - k * (15 // k)
        assert [e["kind"] for e in events[:18]] == ["start"] + ["forward"] * 16 + [
            "backward" if last == 1 else "recompute"]
    assert all(not e["temp"] for e in a.plain)                   # storing all recomputes nothing
    assert "from 17 to 8 at spacings 3 to 6" in a.subtitle_for(2)
    assert "from 17 activations to 8" in a.subtitle_for(3)
    assert "28 times instead of 16" in a.subtitle_for(3)
    np.testing.assert_array_equal(a.plain_grad, s.panels[2].data["values"])
    np.testing.assert_array_equal(a.ck_grad, a.plain_grad)


# ---------------------------------------------------------------------- 90
def test_roofline_measure_times_ceilings_and_operations_separately():
    tiny = dict(repeats=3, copy_mb=1, peak_sizes=(8, 12), matmul=16, attention=(16, 4),
                layernorm=(16, 8), decode=(32, 8))
    d = S.roofline_measure(tiny)
    assert np.shape(d["copy_seconds"]) == (3,) and np.shape(d["peak_seconds"]) == (2, 3)
    assert np.shape(d["seconds"]) == (4, 3) and np.min(d["seconds"]) > 0
    assert d["names"] == ["matmul", "attention", "layer norm", "decode"]
    assert d["copy_bytes"] == 2 * (1 << 20)
    assert d["flops"][0] == 2 * 16 ** 3 and d["bytes"][0] == 3 * 16 * 16 * 8
    assert d["flops"][3] == 2 * 32 * 8 and d["bytes"][3] == (32 * 8 + 32 + 8) * 8


def test_roofline_ceilings_are_measured_apart_from_the_operations():
    s = study("roofline")
    count, speed, roof, took = (p.data for p in s.panels)
    repeats = s.metadata["repeats"]
    assert repeats >= 15 and len(roof["copy_rates"]) == len(roof["peak_samples"]) == repeats
    # each ceiling is the median of its own benchmark's timings, with its spread
    bandwidth, peak = roof["bandwidth"], roof["peak"]
    assert bandwidth[0] == pytest.approx(np.median(roof["copy_rates"]))
    assert peak[0] == pytest.approx(np.median(roof["peak_samples"]))
    assert bandwidth[1] == pytest.approx(roof["copy_rates"].min())
    assert peak[2] == pytest.approx(roof["peak_samples"].max())
    assert bandwidth[1] <= bandwidth[0] <= bandwidth[2] and peak[1] <= peak[0] <= peak[2]
    np.testing.assert_allclose(roof["ys"][0], np.minimum(peak[0], bandwidth[0] * roof["x"]))
    # the benchmarks are not the operations: other shapes, and no operation sits on the roof
    # by construction
    assert S.ROOF["matmul"] not in S.ROOF["peak_sizes"]
    assert 3 * int(roof["peak_size"]) ** 2 * 8 < 8 * 2 ** 20 < count["bytes"].min()
    np.testing.assert_allclose(speed["ceiling"],
                               np.minimum(peak[0], bandwidth[0] * count["values"]))
    np.testing.assert_allclose(speed["reach"], speed["values"] / speed["ceiling"])
    assert not np.isclose(speed["reach"], 1.0, atol=1e-6).any()
    assert (speed["low"] <= speed["values"]).all() and (speed["values"] <= speed["high"]).all()
    np.testing.assert_allclose(count["flops"] / (took["values"] * 1e-6) / 1e9, speed["values"])
    # what the model says of these four: two do far more arithmetic per byte than the ridge
    names = list(count["labels"])
    assert names == ["matmul", "attention", "layer norm", "decode"]
    ridge = peak[0] / bandwidth[0]
    intensity = dict(zip(names, count["values"]))
    assert intensity["matmul"] > intensity["attention"] > 3 * ridge
    assert ridge > 1.5 * intensity["layer norm"] > 1.5 * intensity["decode"]
    # NumPy's layer norm makes several passes over memory: well under its roof
    reach = dict(zip(names, speed["reach"]))
    assert reach["layer norm"] < 0.5
    assert dict(zip(names, speed["values"]))["matmul"] > 3 * dict(
        zip(names, speed["values"]))["layer norm"]


def test_roofline_numbers_are_the_studys():
    s = study("roofline")
    a = M.Roofline(s)
    count, speed, roof, _ = (p.data for p in s.panels)
    assert [op[0] for op in a.ops] == list(count["labels"])
    np.testing.assert_allclose([op[3] / op[4] for op in a.ops], count["values"])
    assert (a.peak, a.bandwidth) == (float(roof["peak"][0]), float(roof["bandwidth"][0]))
    assert "%s operations per byte" % M.number(count["values"].max()) in a.subtitle_for(0)
    assert "timed %d times" % s.metadata["repeats"] in a.subtitle_for(1)
    assert M.number(a.bandwidth) + " GB a second" in a.subtitle_for(2)
    assert M.number(a.peak) + " billion a second" in a.subtitle_for(3)
    for name, share in zip(a.names, speed["reach"]):
        assert "%s %.0f%%" % (name, 100 * share) in a.subtitle_for(4)

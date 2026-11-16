"""Animations 75-80: what they show is what the studies measured, and what
they recompute (the weights after every update, the batch gradients) reproduces
the studies' own curves. 75, 76 and 80 draw the redesigned experiments of
voxanim.study_optim: their claims are tested here too, as relative claims."""

import numpy as np
import pytest
from conftest import data_marks

from voxanim import mech_optim as M
from voxanim import presets, qa
from voxanim import study_optim as O
from voxanim.next_catalog import build_study

NAMES = sorted(M.ANIMATIONS)


def marked(names=NAMES):
    return [pytest.param(n, marks=data_marks(n)) for n in names]


def study(name):
    return build_study("next_systems", name)


def anim(name):
    return presets.get(name).build()


def data(name, panel, key="ys"):
    return np.asarray(study(name).panels[panel].data[key], float)


# ---------------------------------------------------------------------- every one
@pytest.mark.parametrize("name", marked())
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", marked())
def test_is_a_mechanism_from_this_module_and_loops(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45 <= a.t_done <= 60
    first, last = a.scene(0.0), a.scene(a.duration)
    assert [t.s for t in first.texts if t.opacity > 0.02] == \
        [t.s for t in last.texts if t.opacity > 0.02]
    for k in range(len(a.PHASES)):
        assert a.subtitle_for(k)


# ---------------------------------------------------------------------- 75
NEWTON = "newton-vs-gradient-descent"


@pytest.mark.parametrize("name", marked([NEWTON]))
def test_newton_numbers_are_the_studys(name):
    a, s = anim(name), study(name)
    counts, cond = data(name, 2).T, data(name, 2, "x")
    main = list(O.NEWTON_RIDGES).index(s.metadata["ridge_drawn"])
    assert "%d updates to get within 1%%" % counts[main, 0] in a.subtitle_for(1)
    assert "%d updates; but" % counts[main, 1] in a.subtitle_for(2)
    assert "%.4f" % data(name, 1)[3, -1] in a.subtitle_for(3)
    assert "%d updates where that is %.0f" % (counts[-1, 0], cond[-1]) in a.subtitle_for(4)
    lam = np.linalg.eigvalsh(data(name, 0, "values"))
    assert cond[main] == pytest.approx(lam[-1] / lam[0])
    assert "%.0f times" % cond[main] in a.subtitle_for(0)
    assert s.metadata["designs_tried"]


@pytest.mark.parametrize("name", marked([NEWTON]))
def test_newton_study_shows_what_the_step_size_and_the_condition_number_do(name):
    d = O.newton_data()
    counts, cond, loss = d["counts"], d["cond"], d["loss"]
    assert list(d["factors"]) == [1.0, 1.9, 2.1] and (np.diff(cond) < 0).all()
    # Newton's step: one update, to the exact minimum (nothing is lower, and the slope is 0)
    p = O.newton_problem()
    best = d["w"][3, 1]
    assert (counts[:, 2] == 1).all() and loss[3, 1] == pytest.approx(float(d["floor"]))
    assert np.abs(p["h"] @ best - p["b"]).max() < 1e-12
    # gradient descent needs more the worse the conditioning: 130 times more over the range
    assert (np.diff(counts[:, 0]) < 0).all()
    assert counts[0, 0] > 50 * counts[-1, 0] and counts[1, 0] > 50 * counts[1, 2]
    # near the limit it zig-zags yet needs fewer updates where the bowl is long
    assert (counts[:2, 1] < 0.7 * counts[:2, 0]).all()
    # past the limit the loss grows without end; under it, it only falls
    assert loss[2, -1] > 1000 * loss[2, 0] and (np.diff(loss[2, 5:]) > 0).all()
    assert (np.diff(loss[:2], axis=1) < 0).all()
    # the counts are the first update at which the recorded loss is within 1% of the gap
    main = list(O.NEWTON_RIDGES).index(0.02)
    for m in range(2):
        assert counts[main, m] == np.argmax(loss[m] <= d["goal"]) or counts[main, m] > 80
    assert counts[main, 1] == np.argmax(loss[1] <= d["goal"])


@pytest.mark.parametrize("name", marked([NEWTON]))
def test_newton_data_and_route_are_exact(name):
    d, p, _s = O.newton_data(), O.newton_problem(), study(name)
    train, test = p["train"], p["test"]
    assert train["official_partition"] == "train" and test["official_partition"] == "test"
    assert len(set(train["example_indices"])) == 1024 and len(set(test["example_indices"])) == 384
    np.testing.assert_allclose(d["h"], p["a"].T @ p["a"] / 1024 + 0.02 * np.eye(50))
    for m, t in ((0, 7), (1, 40), (3, 1)):                       # the recorded losses, again
        e = p["a"] @ d["w"][m, t] - p["target"]
        want = np.mean(np.sum(e * e, axis=1)) / 2 + 0.01 * np.sum(d["w"][m, t] ** 2)
        assert d["loss"][m, t] == pytest.approx(want)
    held = np.mean((p["at"] @ d["w"][3, 1]).argmax(1) == p["yt"])
    assert data(name, 3, "values")[2] == pytest.approx(held)
    a = anim(name)
    lam = a.lam
    # the route drawn: each coordinate shrinks by its own exact factor 1 - step x curvature
    np.testing.assert_allclose(a.route[3, 1:], 0.0, atol=1e-9)
    for m, factor in enumerate(O.NEWTON_STEPS):
        shallow, steep = a.route[m, :, 0], a.route[m, :, 1]
        np.testing.assert_allclose(shallow[1:] / shallow[:-1], 1 - factor * lam[1] / lam[0],
                                   rtol=1e-6)
        if factor != 1.0:                                        # zig-zag: the sign flips
            np.testing.assert_allclose(steep[1:6] / steep[:5], 1 - factor, rtol=1e-6)
    np.testing.assert_allclose(a.route[0, 1:, 1], 0.0, atol=1e-9)   # the safe step: the floor
    np.testing.assert_allclose(np.abs(a.slopes[1]) / lam[0], abs(a.start[1]))
    assert 2 < a._inside(2) < 30                                 # 2.1 leaves the map


# ---------------------------------------------------------------------- 79
@pytest.mark.parametrize("name", marked(["second-order-preconditioners"]))
def test_preconditioner_numbers_are_the_studys(name):
    a, cond, loss = anim(name), data(name, 2, "values"), data(name, 1)
    assert "from %.0f to %.0f" % (cond[0], cond[1]) in a.subtitle_for(2)
    assert "the ratio %.0f" % cond[2] in a.subtitle_for(3)
    for v in loss[:, -1]:
        assert "%.4f" % v in a.subtitle_for(4)
    for v in data(name, 3, "values"):
        assert "%.1f%%" % (100 * v) in a.subtitle_for(4)
    np.testing.assert_allclose(M.ridge_paths(False)["loss"], loss, rtol=0, atol=1e-12)
    # the matrices drawn have the conditioning the study reports
    for mat, value in zip(a.mats, cond):
        np.testing.assert_allclose(np.linalg.cond(mat), value, rtol=1e-6)


@pytest.mark.parametrize("name", marked(["second-order-preconditioners"]))
def test_coupled_pair_is_exact(name):
    pair, h = M.coupled_pair(), M.ridge_problem()["h"]
    i, j = pair["pixels"]
    np.testing.assert_allclose(pair["h"], h[np.ix_([i, j], [i, j])])
    d = np.sqrt(np.diag(h))
    r = (h / np.outer(d, d))[:49, :49]
    np.fill_diagonal(r, 0)
    assert pair["rho"] == pytest.approx(r.max())
    best, paths = pair["best"], pair["paths"]
    np.testing.assert_allclose(paths[2, 1], best)                 # the full solve: one step
    cost = np.einsum("mti,ij,mtj->mt", paths - best, pair["h"], paths - best)
    assert (np.diff(cost[:2], axis=1) < 0).all()                 # the other two only descend
    assert cost[0, -1] > 1e-6 and cost[1, -1] > 1e-6             # and are not there after 30
    # in each method's coordinates the curvature is what the picture shows
    a = anim(name)
    for t, want in zip(a.maps, (pair["h"], pair["h"] / np.outer(*[np.sqrt(np.diag(pair["h"]))] * 2),
                                np.eye(2))):
        back = np.linalg.inv(t)
        np.testing.assert_allclose(back.T @ pair["h"] @ back, want, atol=1e-12)


# ---------------------------------------------------------------------- 76
EDGE = "edge-of-stability"


@pytest.mark.parametrize("name", marked([EDGE]))
def test_edge_numbers_are_the_studys(name):
    a, s = anim(name), study(name)
    loss, sharp, level = data(name, 0), data(name, 1), data(name, 3, "each")
    assert "from %.2f to the limit 2 ÷ 0.3 = 6.67" % sharp[1, 0] in a.subtitle_for(1)
    assert "%.4f against %.4f" % (loss[1, -1], loss[0, -1]) in a.subtitle_for(3)
    assert "averages %.2f, %.2f and %.2f" % tuple(level.mean(0)[1:]) in a.subtitle_for(4)
    w = data(name, 2)
    assert "from %.3f to %.3f" % (w[0, 0], w[0].max()) in a.subtitle_for(2)
    np.testing.assert_allclose(data(name, 3, "values"),
                               (level * np.array(O.EDGE_STEPS) / 2).mean(0))
    np.testing.assert_allclose(data(name, 1, "limits"), 2 / np.array(O.EDGE_STEPS))
    assert len(s.metadata["designs_tried"]) == 3 and s.metadata["held_out_accuracy"]


@pytest.mark.parametrize("name", marked([EDGE]))
def test_edge_study_shows_the_curvature_rising_to_each_limit(name):
    d = O.edge_data()
    sharp, loss, level = d["sharp"], d["loss"], O.edge_levels(d)
    limit = 2 / np.array(O.EDGE_STEPS)
    start = sharp[:, :, 0]
    # every run starts far under every limit (the first study started above one)
    assert (start < 0.7 * limit.min()).all()
    # 0.3, 0.4 and 0.5: the curvature climbs to 2 / step size and stays there, in all 3 starts
    assert (np.abs(level[:, 1:] / limit[1:] - 1) < 0.02).all()
    assert (level[:, 1:] > 1.5 * start[:, 1:]).all()
    late = sharp[:, 1:, O.EDGE_SETTLED // O.EDGE_EVERY:]
    assert (np.abs(late / limit[1:, None] - 1) < 0.08).all()      # hovering, not passing by
    # the control, 0.2: its limit is never reached and its loss never rises
    assert (sharp[:, 0].max(-1) < 0.9 * limit[0]).all()
    rises = (np.diff(loss, axis=-1) > 0).sum(-1)
    assert (rises[:, 0] == 0).all() and (rises[:, 1:] > 500).all()
    # at the edge the loss rises hundreds of times and still ends lower than the control's
    assert (loss[:, 1:, -1] < 0.9 * loss[:, :1, -1]).all()
    assert (loss[:, :, -1] < 0.01 * loss[:, :, 0]).all()
    # held out: a disjoint 1,000 test digits; every step size reads about as many of them
    acc = d["held"][..., 1]
    assert acc.min() > 0.7 and acc.max() - acc.min() < 0.03


@pytest.mark.parametrize("name", marked([EDGE]))
def test_edge_measurements_are_what_they_say(name):
    d, split = O.edge_data(), O.edge_data_split()
    assert split["train"]["official_partition"] == "train"
    assert split["test"]["official_partition"] == "test"
    assert split["x"].shape == (256, 196) and abs(split["x"].mean(0)).max() < 1e-12
    P = O.edge_init(O.EDGE_SEEDS[0])
    assert d["loss"][0, 0, 0] == pytest.approx(O.edge_loss(P, split["x"], split["t"], False))
    # the curvature: a long power iteration from another start finds the same eigenvalue
    v = np.random.default_rng(5).normal(size=len(O.edge_pack(P)))
    lam, v = O.edge_sharpness(P, split["x"], split["t"], v, 150)
    assert d["sharp"][0, 0, 0] == pytest.approx(lam, rel=0.01)
    # the valleys drawn bend as much as the curvature measured there says
    width, mid = d["width"], len(d["width"]) // 2
    for line, lam in zip(d["profiles"][1:], d["snap_sharp"][1:]):
        bend = (line[mid + 1] - 2 * line[mid] + line[mid - 1]) / (width[1] - width[0]) ** 2
        assert bend == pytest.approx(lam, rel=0.05)
    # the excursion: the weights' place in the valley is where the loss was recorded...
    for line, off, value in zip(d["w_line"], d["w_off"], d["w_loss"]):
        assert np.interp(off, d["w_grid"], line) == pytest.approx(value, rel=0.02, abs=1e-4)
    # ...they change side at every update, by the factor 1 - step size x curvature while the
    # swing is small, growing while that is past 2 and shrinking once it is under
    off, times = d["w_off"], O.EDGE_STEPS[O.EDGE_MAIN] * d["w_lam"]
    assert (off[1:] * off[:-1] < 0).all()
    np.testing.assert_allclose(off[1:16] / off[:15], 1 - times[:15], rtol=0.03)
    peak = int(np.abs(off).argmax())
    assert (times[:peak - 4] > 2).all() and (np.diff(np.abs(off[:peak - 2])) > 0).all()
    assert (times[peak + 4:] < 2).all() and (np.diff(np.abs(off[peak + 2:])) < 0).all()
    assert d["w_loss"].max() > 5 * d["w_loss"][0]
    a = anim(name)
    assert a.q1 == pytest.approx(a.q0 * (1 - a.lr * a.snap_sharp[0]), abs=0.02)


# ---------------------------------------------------------------------- 77
@pytest.mark.parametrize("name", marked(["batch-size-and-noise"]))
def test_batch_numbers_are_the_studys(name):
    a, noise, loss = anim(name), data(name, 0)[0], data(name, 2)
    assert "%.2f with 1 digit" % noise[0] in a.subtitle_for(1)
    assert "%.3f with 256" % noise[4] in a.subtitle_for(1)
    assert "(%.3f)" % data(name, 3, "values")[0] in a.subtitle_for(2)
    for v in loss[:, -1]:
        assert "%.2f" % v in a.subtitle_for(3)
    assert a.sizes == [1, 4, 16, 64, 256] and a.sizes[a.match] == 64


@pytest.mark.parametrize("name", marked(["batch-size-and-noise"]))
def test_batch_gradients_are_the_studys_batches(name):
    g = M.batch_gradients()
    np.testing.assert_allclose(g["noise"], data(name, 0)[0], rtol=1e-10)
    signal = data(name, 3, "values")[0]
    np.testing.assert_allclose(g["signal"], signal, rtol=1e-10)
    np.testing.assert_allclose(g["full"], [np.sqrt(signal), 0.0], atol=1e-9)
    assert g["xy"].shape == (5, 100, 2)
    assert (g["inplane"] < g["noise"]).all() and (np.diff(g["inplane"]) < 0).all()
    for k, b in enumerate(g["batches"]):
        picks = g["index"][k]
        assert ((picks[:, :b] >= 0) & (picks[:, :b] < 512)).all() and (picks[:, b:] == -1).all()


# ---------------------------------------------------------------------- 78
@pytest.mark.parametrize("name", marked(["weight-averaging"]))
def test_averaging_numbers_are_the_studys(name):
    a, loss = anim(name), data(name, 1)
    for v in loss[:, -1]:
        assert "%.3f" % v in a.subtitle_for(3)
    for v in data(name, 3, "values"):
        assert "%.2f" % v in a.subtitle_for(4)


@pytest.mark.parametrize("name", marked(["weight-averaging"]))
def test_averaging_rerun_is_the_studys_run(name):
    r = M.averaging_run()
    # SGD's every fifth set of weights, in the study's plane (a direction's sign is arbitrary)
    np.testing.assert_allclose(np.abs(r["xy"][0, ::5]), np.abs(data(name, 0, "xy")), atol=1e-8)
    np.testing.assert_allclose(r["final"][:, 0], data(name, 1)[:, -1], atol=1e-12)
    np.testing.assert_allclose(r["final"][:, 1], data(name, 2)[:, -1], atol=1e-12)
    late = r["late"]
    np.testing.assert_allclose(late[2, -1], 0.0, atol=1e-9)       # the plane's origin: the mean
    assert abs(late[0, -1, 1]) < 1e-9                             # SGD's end: on its first axis
    np.testing.assert_allclose(late[0, 0], late[2, 0])            # the mean starts as SGD
    # the mean of the SGD weights visited after update 200 (linear, so also in the plane)
    np.testing.assert_allclose(late[2, -1], late[0, 1:].mean(0), atol=1e-9)
    # the exponential average: 3% of the way to the new weights, every update
    np.testing.assert_allclose(late[1, 1:], 0.97 * late[1, :-1] + 0.03 * late[0, 1:], atol=1e-9)
    # the three final models lie in the plane: the loss drawn under them is their own
    for (x, y), want in zip(late[:, -1], r["final"][:, 0]):
        i, j = np.abs(r["fy"] - y).argmin(), np.abs(r["fx"] - x).argmin()
        assert abs(r["field"][i, j] - want) < 0.02
    assert r["picks"].shape == (400, 32)


# ---------------------------------------------------------------------- 80
SPIKES = "loss-spikes"


@pytest.mark.parametrize("name", marked([SPIKES]))
def test_spike_numbers_are_the_studys(name):
    a, s = anim(name), study(name)
    loss, worst = data(name, 0), data(name, 3, "each")
    at, m = s.metadata["spike_drawn_at"], a.m
    assert "from %.2f to %.3f" % (loss[0, 0], loss[0, 300]) in a.subtitle_for(0)
    assert "to %.2f, on unchanged data" % loss[0, at] in a.subtitle_for(2)
    assert "%.1f times its lowest" % worst[1, m] in a.subtitle_for(3)
    assert "%.3f against %.3f" % tuple(data(name, 3, "tail").mean(1)[[1, 0]]) in a.subtitle_for(4)
    spiked = (data(name, 3, "count") > 0).sum(1)
    assert "plain Adam spikes in %d" % spiked[0] in a.subtitle_for(4)
    assert "the larger ε in %d and clipping in %d" % (spiked[2], spiked[3]) in a.subtitle_for(4)
    np.testing.assert_allclose(data(name, 3, "values"), worst.mean(1))
    assert len(a.rows[0]) == 13 and len(s.metadata["designs_tried"]) >= 3


@pytest.mark.parametrize("name", marked([SPIKES]))
def test_spike_study_spikes_by_itself_and_a_short_memory_stops_it(name):
    d = O.spike_data()
    s = O.spike_summary(d)
    worst, count = s["worst"], s["count"]
    # plain Adam, fixed data: the loss leaps to over 10 times its lowest in most starts
    assert (worst[0] > 10).sum() >= 2 and (count[0] > 0).sum() >= 2
    # a memory of 10 updates: never twice its lowest, in any start; 20 times less on average
    assert (worst[1] < 2).all() and (count[1] == 0).all()
    assert worst[1].mean() < worst[0].mean() / 20
    # ...at a price: a higher loss at the end than plain Adam's best
    assert s["tail"][1].min() > s["tail"][0].min() + 0.001
    # the larger epsilon and clipping are recorded as they came out: neither is spike-free
    assert (count[2] > 0).any() and (count[3] > 0).any()
    # inside the run drawn: the memory fades and the step per unit of gradient grows first
    a = anim(name)
    assert a.fade > 1.5 and a.grow > 1.3
    assert a.kick > 10 and a.leap > 10
    before = d["loss"][0, a.m, a.q0:a.q1]
    assert before.max() < 1.5 * before.min()                      # quiet until then
    assert d["loss"][0, a.m, a.at] > 10 * before.min()
    assert d["moment"][0, a.m, a.at - 3] < 1.2 * d["moment"][0, a.m, a.q1]   # memory lags


@pytest.mark.parametrize("name", marked([SPIKES]))
def test_spike_run_is_recorded_as_it_ran(name):
    from voxanim import realdata as R
    from voxanim import tinylm as T

    d = O.spike_data()
    seq, held = O.spike_text()
    np.testing.assert_array_equal(d["text"], seq)
    assert seq.shape == (64, 13) and held.shape == (256, 13)
    assert R.text_metadata("train")["character_range"][1] <= \
        R.text_metadata("test")["character_range"][0]             # held out: later text
    assert d["loss"].shape == (4, 3, O.SPIKE_UPDATES) and (d["step"] > 0).all()
    np.testing.assert_allclose(d["used"][:3], d["norm"][:3])      # only the last one clips
    np.testing.assert_allclose(d["used"][3], np.minimum(d["norm"][3], 1.0))
    assert (d["held"] > 10 * d["loss"][:, :, -1]).all()           # memorised, and said so
    # the first updates of plain Adam, again: the loss, the gradient and the step recorded
    cfg = T.Config(vocab=len(R.vocabulary()), width=16, heads=2, kv_heads=2, ff=24)
    p = T.init(cfg, O.SPIKE_SEEDS[0])
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(q) for k, q in p.items()}
    for i in range(3):
        loss, g = T.loss_grads(p, seq[:, :-1], seq[:, 1:], cfg)
        assert d["loss"][0, 0, i] == pytest.approx(loss, rel=1e-9)
        assert d["norm"][0, 0, i] == pytest.approx(
            np.sqrt(sum(float((z * z).sum()) for z in g.values())), rel=1e-9)
        moved = 0.0
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k]
            step = O.SPIKE_RATE * (m[k] / (1 - 0.9 ** (i + 1))) / (
                np.sqrt(v[k] / (1 - 0.999 ** (i + 1))) + 1e-8)
            p[k] -= step
            moved += float((step * step).sum())
        assert d["step"][0, 0, i] == pytest.approx(np.sqrt(moved), rel=1e-9)
    events, worst = O.spike_events(np.array([5.0, 1.0, 1.0, 3.0, 1.0, 0.5, 1.2]), start=2)
    assert list(events) == [3, 6] and worst == pytest.approx(3.0)

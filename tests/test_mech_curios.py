"""Animations 97-100 (voxanim.mech_curios): the layout is clean and the numbers on
screen are the study's. For 97-99 (voxanim.study_curios, the redesigned experiments)
the claims each study now supports are held as relative claims with a margin, and
what the drawings show is checked against the rules that produced it; for 100 the
module's replay is the study's own run."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import available, data_marks

from voxanim import mech_curios as M
from voxanim import next_systems as S
from voxanim import presets, qa
from voxanim import study_curios as SC
from voxanim.next_catalog import build_study

NAMES = ("hopfield-network", "neural-cellular-automata", "neural-ode", "evolution-strategies")


def marked(name):
    return pytest.param(name, marks=data_marks(name))


def needs(name):
    if not available(name):
        pytest.skip("data for %s not downloaded" % name)


def study(name):
    return build_study("next_systems", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


@pytest.mark.parametrize("name", [marked(n) for n in NAMES])
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", [marked(n) for n in NAMES])
def test_drawn_by_this_module_and_loop_closes(name):
    anim = presets.get(name).build()
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    middle = anim.scene(anim.spans[1][0] + 3.0)
    assert len(middle.quads) >= 3                    # something is drawn, not only text


# ---------------------------------------------------------------------- 97
def test_hopfield_claims():
    needs("hopfield-network")
    s, d = study("hopfield-network"), SC.hopfield_data()
    pat = d["patterns"]
    rate = dict(zip(SC.HOP_RULES, s.panels[2].data["ys"]))
    at = {k: i for i, k in enumerate(SC.HOP_LOADS)}
    # the honest contrast: the Hebbian rule lands on a blend, lower than every memory ...
    off = s.metadata["demo_pixels_off"]
    assert off["hebbian"] > 40 and d["sweeps"][-1] < d["memory_energy"].min()
    # ... and the three remedies bring the same corrupted digit back
    assert max(off["centred"], off["pseudo-inverse"], off["modern"]) <= SC.HOP_TOLERANCE
    # capacity, measured: every rule holds two digits; raw Hebbian fails at three
    assert min(r[at[2]] for r in rate.values()) >= 0.95
    assert rate["hebbian"][at[3]] < rate["centred"][at[3]] - 0.5
    assert rate["hebbian"][at[3]:].max() < 0.1
    # centring buys a few digits, the pseudo-inverse rule dozens, the modern rule more
    assert rate["centred"][at[8]] < rate["pseudo-inverse"][at[8]] - 0.5
    assert rate["pseudo-inverse"][at[64]] >= 0.95
    assert rate["pseudo-inverse"][at[256]] < rate["modern"][at[256]] - 0.5
    # corruption: with 4 stored the centred rule fades as more pixels flip
    flips = dict(zip(SC.HOP_RULES, s.panels[3].data["ys"]))
    assert flips["centred"][-1] < flips["centred"][0] - 0.3
    assert flips["pseudo-inverse"][:7].min() >= 0.95 and flips["hebbian"].max() < 0.1
    assert np.array_equal(pat[1], d["pinv_final"])


def test_hopfield_data_is_what_it_says():
    needs("hopfield-network")
    from voxanim import realdata
    from voxanim.mnist import load

    s, d = study("hopfield-network"), SC.hopfield_data()
    x, y, _, _ = load()
    assert list(y[d["demo_indices"]]) == [0, 1, 2]
    assert np.array_equal(d["patterns"], np.where(x[d["demo_indices"]] > 0.3, 1.0, -1.0))
    pool = d["pool_indices"]
    assert len(set(pool.tolist())) == len(pool) == 4096 and pool.max() < 50000
    assert d["load_wrong"].shape == (len(SC.HOP_LOADS), 4, SC.HOP_TRIALS)
    assert d["flip_wrong"].shape == (2, len(SC.HOP_FLIPS), 4, SC.HOP_TRIALS)
    assert np.allclose(s.panels[2].data["ys"], (d["load_wrong"] <= SC.HOP_TOLERANCE).mean(-1).T)
    assert np.allclose(s.panels[3].data["ys"],
                       (d["flip_wrong"][0] <= SC.HOP_TOLERANCE).mean(-1).T)
    # one cell of the capacity count, run again from the stored digits
    px, _, _ = realdata.mnist_split(4096, "train", 28, SC.SEED)
    again = SC._hop_cell(np.where(px > 0.3, 1.0, -1.0), 4, SC.HOP_FLIP, [SC.SEED, 2, 4])
    assert np.array_equal(again, d["load_wrong"][SC.HOP_LOADS.index(4)])
    # the recall the animation follows: each vote is the field the pixel saw, the state
    # follows it, and the energy never rises
    before = np.vstack([d["start"], d["states"][:-1]])
    j = d["order"]
    assert np.allclose(d["votes"], np.einsum("nk,nk->n", d["W"][j], before))
    moved = d["votes"] != 0
    assert np.array_equal(d["states"][np.arange(784), j][moved], np.sign(d["votes"][moved]))
    assert np.all(np.diff(d["energy"]) <= 1e-9)
    assert np.isclose(d["energy"][-1], d["sweeps"][1]) and np.isclose(d["sweeps"][1],
                                                                      d["sweeps"][-1])
    assert np.array_equal(d["states"][-1], d["final"])
    shown = (s.panels[0].data["values"] * 2 - 1).reshape(8, 784)
    assert np.array_equal(shown[:3], d["patterns"]) and np.array_equal(shown[3], d["start"])
    assert np.array_equal(shown[4], d["final"]) and np.array_equal(shown[6], d["pinv_final"])
    # the remedies settle within the sweep the animation shows
    for key in ("centred", "pinv"):
        assert np.array_equal(d[key + "_states"][-1], d[key + "_final"])
    # the pseudo-inverse weights hold every stored digit exactly (before the diagonal is cleared)
    P = d["patterns"]
    assert np.allclose(np.linalg.pinv(P) @ P @ P.T, P.T, atol=1e-8)


def test_hopfield_numbers_on_screen():
    needs("hopfield-network")
    s = study("hopfield-network")
    anim = M.HopfieldNetwork(s)
    sub = subtitles(anim)
    energy = s.panels[1].data["ys"][0]
    assert "from %.0f to %.0f" % (energy[0], energy[-1]) in sub[1]
    assert "%d of the 784" % int((anim.d["start"] != anim.d["patterns"][1]).sum()) in sub[1]
    final = (s.panels[0].data["values"][4] * 2 - 1).ravel()
    assert anim.differ[1] == int((final != anim.d["patterns"][1]).sum())
    assert "%d pixels away from the stored 1" % anim.differ[1] in sub[2]
    off = s.metadata["demo_pixels_off"]
    assert "with %d and %d pixels wrong" % (off["centred"], off["pseudo-inverse"]) in sub[3]
    assert "%.2f, %.2f, %.2f" % tuple(anim.d["prob"]) in sub[4]
    assert anim.blend_differ == 0                                 # "the 1 comes back"
    largest = s.metadata["largest_load_recalled_9_in_10"]
    assert "up to %d digits (Hebbian), %d (minus the mean), %d (pseudo-inverse)" % (
        largest["hebbian"], largest["centred"], largest["pseudo-inverse"]) in sub[5]
    assert "%d in 100 at 256" % round(100 * s.panels[2].data["ys"][3, -1]) in sub[5]
    # the pair of pixels phase 1 follows: the weight shown is the matrix's entry
    for i in anim.partners:
        agree = int((anim.d["patterns"][:, i] * anim.d["patterns"][:, anim.j]).sum())
        assert np.isclose(anim.d["W"][anim.j, i], agree / 784)


# ---------------------------------------------------------------------- 98
def test_cellular_claims():
    needs("neural-cellular-automata")
    s = study("neural-cellular-automata")
    e = s.panels[3].data["all_errors"]                 # (rule, run, step)
    n, cut = len(SC.SEEDS), SC.NCA_CUT
    pool, only = e[:n], e[n:]
    start = e[0, 0, 0]
    # every rule grows the digit
    assert e[:, 0, cut - 1].max() < 0.01 * start
    # the pool-trained rules hold it for 960 steps; the seed-only rules lose it
    assert pool[:, 0, -1].max() < 0.01 * start
    assert only[:, 0, -1].min() > 20 * pool[:, 0, -1].max()
    assert only[:, 0, -1].min() > 20 * only[:, 0, cut - 1].max()
    # all three damages grow back within 48 steps, on every seed; not without the pool
    hurt, mended = pool[:, 1:, cut], pool[:, 1:, 2 * cut]
    assert hurt.min() > 0.3 * start and mended.max() < 0.01 * hurt.min()
    assert only[:, 1:, 2 * cut].min() > 50 * mended.max()
    assert pool[:, 1:, -1].max() < 0.01 * hurt.min()               # and stay mended
    # training: the loss fell on every rule
    loss = s.panels[1].data["ys"]
    assert (loss[:, -50:].mean(1) < 0.05 * loss[:, :10].mean(1)).all()


def test_cellular_data_is_what_it_says():
    needs("neural-cellular-automata")
    from voxanim.mnist import load

    s, d = study("neural-cellular-automata"), SC.nca_data()
    x, y, _, _ = load()
    index = int(d["target_index"])
    assert y[index] == 0
    assert np.allclose(d["target"], x[index].reshape(14, 2, 14, 2).mean((1, 3)))
    states = d["states"]
    assert states.shape == (2, 4, SC.NCA_KEEP + 1, 14, 14, SC.NCA_CHANNELS)
    # the kept weights are the rule: one step of it gives the next recorded state
    masks = SC.nca_eval_masks(SC.NCA_KEEP)
    P = dict(A=d["A"], b=d["b"], B=d["B"])
    for which, rule in ((0, 0), (1, len(SC.SEEDS))):
        for t in (0, 20, 100):
            nxt = SC.nca_step(SC.nca_one(P, rule), states[which][None, :, t], masks[t])[0]
            assert np.allclose(nxt[0], states[which][:, t + 1], atol=1e-4)
    # the cuts are the cuts, made at the step the study says
    cut = SC.NCA_CUT
    assert np.array_equal(states[0, 0, :cut], states[0, 1, :cut])
    assert not states[0, 1, cut, 3:9, 3:9].any() and not states[0, 2, cut, :, 7:].any()
    assert not states[0, 3, cut, 7:].any() and states[0, 3, cut, :7].any()
    for run in (1, 2, 3):
        assert np.array_equal(states[0, run, cut], SC.nca_damage(states[0, 0, cut], run - 1))
    # the errors are of those states, and the panels show them
    e = s.panels[3].data["all_errors"]
    assert np.allclose(((states[0, ..., 0] - d["target"]) ** 2).mean((2, 3)),
                       e[0, :, :SC.NCA_KEEP + 1], atol=1e-6)
    assert np.allclose(s.panels[0].data["values"][5], states[0, 0, cut - 1, ..., 0])
    assert np.allclose(d["snaps"][-1], states[0, 0, cut, ..., 0], atol=1e-4)
    assert not d["snaps"][0].any()                     # before training nothing grows
    # backpropagation through a rollout, against finite differences
    rng = np.random.default_rng(0)
    Q = dict(A=rng.normal(0, 0.4, (2, 12, 6)), b=rng.normal(0, 0.1, (2, 1, 6)),
             B=rng.normal(0, 0.3, (2, 6, 4)))
    start = rng.normal(0, 1, (2, 3, 14, 14, 4))
    fire = (rng.random((3, 2, 3, 14, 14)) < 0.5).astype(float)
    _, G, _ = SC.nca_loss(Q, start, fire, d["target"])
    for key in Q:
        i = tuple(int(rng.integers(0, size)) for size in Q[key].shape)
        up, down = ({k: v.copy() for k, v in Q.items()} for _ in range(2))
        up[key][i] += 1e-6
        down[key][i] -= 1e-6
        slope = (SC.nca_loss(up, start, fire, d["target"])[0].mean(1).sum()
                 - SC.nca_loss(down, start, fire, d["target"])[0].mean(1).sum()) / 2e-6
        assert np.isclose(G[key][i], slope, rtol=1e-4, atol=1e-8)


def test_cellular_numbers_on_screen():
    needs("neural-cellular-automata")
    s = study("neural-cellular-automata")
    anim = M.NeuralCellularAutomata(s)
    sub = subtitles(anim)
    e = s.panels[3].data["all_errors"]
    n, cut = len(SC.SEEDS), SC.NCA_CUT
    C, H = SC.NCA_CHANNELS, SC.NCA_HIDDEN
    assert anim.weights == 3 * C * H + H + H * C and "%d weights" % anim.weights in sub[0]
    assert "step %d its error against the target is %s" % (cut - 1, M.small(e[0, 0, cut - 1])) \
        in sub[1]
    assert "for %d updates" % SC.NCA_UPDATES in sub[2]
    assert "error %s at most" % M.small(e[:n, 0, -1].max()) in sub[3]
    assert "drifts to %s, %s and %s" % tuple(M.small(v) for v in e[n:, 0, -1]) in sub[3]
    assert "%d times the longest" % (SC.NCA_RUN // SC.NCA_STEPS[1]) in sub[3]
    assert "from %.3f, %.3f and %.3f to %s, %s and %s" % (
        tuple(e[0, 1:, cut]) + tuple(M.small(v) for v in e[0, 1:, 2 * cut])) in sub[4]
    assert "worst of three seeds is %s" % M.small(e[:n, 1:, 2 * cut].max()) in sub[4]
    assert M.small(0.0443) == "0.044" and M.small(0.00007) == "0.00007"
    # what phase 1 says the followed cell senses is what the rule is fed
    r, k = anim.CELLS[0]
    state = anim.pool[0, M.NCA_AT]
    dx, dy = SC.nca_sobel(state[None])
    assert np.allclose(anim.sense[r, k], [state[r, k], dx[0, r, k] / 8, dy[0, r, k] / 8])
    assert np.isclose(dx[0, r, k, 0] / 8, (
        state[r - 1:r + 2, k + 1, 0] @ [1, 2, 1] - state[r - 1:r + 2, k - 1, 0] @ [1, 2, 1]) / 8)


# ---------------------------------------------------------------------- 99
def test_ode_claims():
    needs("neural-ode")
    s = study("neural-ode")
    data = s.panels[2].data
    acc, ref, line, gap = data["ys"], data["reference"], float(data["line"][0]), data["gap"]
    steps = list(data["x"])
    never = s.metadata["heldout_accuracy"]["never_answering_5"]
    # a straight line cannot do it; the flow can, on every seed
    assert line < never + 0.02
    assert ref.min() > line + 0.08 and ref.min() > never + 0.08
    # the same weights in fewer, larger Euler steps: further from the flow, and less accurate
    assert (gap[:, -1] < 0.1 * gap[:, 0]).all() and (np.diff(gap, axis=1) < 0).all()
    assert (acc[:, 0] < ref - 0.02).all()                      # one layer costs every seed
    assert acc[0, 0] < ref[0] - 0.2                            # ... and the drawn seed most
    assert (np.abs(acc[:, steps.index(64)] - ref) < 0.01).all()
    # a residual network with a copy of the weights per layer is the Euler run, exactly
    assert s.metadata["resnet_equals_euler_max_difference"] <= 1e-12
    # the adjoint's gradient approaches backpropagation's as the steps shrink
    adjoint = s.panels[3].data["ys"][0]
    assert adjoint[-1] < 0.05 and adjoint[-1] < 0.1 * adjoint[0]
    stored, kept = s.panels[3].data["stored"], s.panels[3].data["kept"][0]
    assert (stored > kept).all() and stored[-1] > 100 * kept


def test_ode_data_is_what_it_says():
    needs("neural-ode")
    s, d = study("neural-ode"), SC.ode_data()
    meta = s.metadata["dataset"]
    assert meta["train"]["official_partition"] == "train"
    assert meta["test"]["official_partition"] == "test"          # disjoint files
    assert set(np.unique(d["test_digits"])) == set(SC.ODE_DIGITS) == set(
        np.unique(d["train_digits"]))
    plane = SC.ode_plane()
    assert np.allclose(plane["test"], d["test"]) and np.allclose(plane["train"], d["train"])
    y = (d["test_digits"] == 5).astype(float)
    P = SC.ode_model(d)
    # the drawn paths are the model's, and the accuracies are of its end points
    drawn = d["test"][:SC.ODE_DRAWN]
    assert np.allclose(SC.ode_euler(P, drawn, 16), d["path_16"])
    assert np.allclose(SC.ode_euler(P, drawn, 1)[-1], drawn + SC.ode_field(P, drawn)[0])
    acc = s.panels[2].data["ys"]
    assert np.isclose(SC.ode_accuracy(P, SC.ode_euler(P, d["test"], 4)[-1], y),
                      acc[0, SC.ODE_STEPS.index(4)])
    exact = SC.ode_rk4(P, d["test"])
    assert np.allclose(exact, d["reference_ends"][0])
    assert np.isclose(SC.ode_accuracy(P, exact, y), s.panels[2].data["reference"][0])
    assert np.allclose(SC.ode_rk4(P, drawn, 128), exact[:SC.ODE_DRAWN], atol=1e-6)
    # the straight-line control is the best line: no nearby line does better on training data
    line = ((d["train"] @ d["line_w"] + d["line_c"]) > 0) == (d["train_digits"] == 5)
    assert np.isclose(line.mean(), d["line_train_accuracy"])
    # the residual stack with separate copies is the Euler run
    layers = [{k: P[k].copy() for k in ("W", "b", "U", "v")} for _ in range(8)]
    assert np.array_equal(SC.ode_resnet(layers, drawn), SC.ode_euler(P, drawn, 8)[-1])
    # backpropagation against finite differences, and the adjoint against backpropagation
    x, t = d["test"][:32], y[:32]
    _, G = SC.ode_backprop(P, x, t, 8)
    rng = np.random.default_rng(0)
    for key in ("W", "b", "U", "v", "w"):
        i = tuple(int(rng.integers(0, size)) for size in P[key].shape)
        up, down = ({k: np.array(v, float) for k, v in P.items()} for _ in range(2))
        up[key][i] += 1e-6
        down[key][i] -= 1e-6
        slope = (SC.ode_backprop(up, x, t, 8)[0] - SC.ode_backprop(down, x, t, 8)[0]) / 2e-6
        assert np.isclose(G[key][i], slope, rtol=1e-4, atol=1e-9)
    exact_gradient = SC.ode_pack(SC.ode_backprop(P, x, t, 4096)[1])
    approx, start = SC.ode_adjoint(P, x, t, 4096)
    assert np.linalg.norm(SC.ode_pack(approx) - exact_gradient) < 0.01 * np.linalg.norm(
        exact_gradient)
    assert np.abs(start - x).max() < 0.05
    # the backward pass drawn is the adjoint's
    back = SC.ode_adjoint(P, drawn, y[:SC.ODE_DRAWN], 16)[1]
    assert np.allclose(back, d["back_16"][-1])


def test_ode_task_was_chosen_on_validation_data():
    needs("neural-ode")
    rows = SC.ode_task_scan()
    gap, digits, _line, _near = rows[0]
    assert digits == SC.ODE_DIGITS and gap > 0.1 and gap > rows[1][0]


def test_ode_numbers_on_screen():
    needs("neural-ode")
    s = study("neural-ode")
    anim = M.NeuralODE(s)
    sub = subtitles(anim)
    data = s.panels[2].data
    acc, ref, line = 100 * data["ys"], 100 * data["reference"], 100 * float(data["line"][0])
    held = s.metadata["heldout_accuracy"]
    assert "only %.1f%%" % line in sub[0]
    assert "never answering 5 (%.1f%%)" % (100 * held["never_answering_5"]) in sub[0]
    assert "gets %.1f%% of the %d" % (ref[0], len(anim.d["test"])) in sub[1]
    assert "%.1f to %.1f" % (ref.min(), ref.max()) in sub[1]
    assert "1 layer get %.1f%%, as 4 %.1f%%, as 16 %.1f%% (exact flow %.1f%%)" % (
        acc[0, 0], acc[0, 2], acc[0, 4], ref[0]) in sub[2]
    assert "%.1f and %.1f points" % (ref[1] - acc[1, 0], ref[2] - acc[2, 0]) in sub[2]
    adjoint = s.panels[3].data["ys"][0]
    assert "%.0f%% at 16 steps and %.1f%% at 512" % (
        100 * adjoint[SC.ODE_ADJOINT_STEPS.index(16)], 100 * adjoint[-1]) in sub[3]
    # the followed point is the one whose one-layer miss is the median, as the module says
    exact = anim.d["reference_ends"][0][:SC.ODE_DRAWN]
    miss = np.linalg.norm(anim.paths[1][-1] - exact, axis=1)
    assert np.sum(miss < miss[anim.focus]) == len(miss) // 2
    assert np.isclose(anim.end_gap[1], miss[anim.focus]) and anim.back_gap > 0
    # the view is the plane turned, not stretched: distances on screen are distances in it
    view = anim.wide
    a, b = np.array([0.3, -1.0]), np.array([1.5, 2.0])
    assert np.isclose(np.linalg.norm(view.px(a) - view.px(b)), view.scale * np.linalg.norm(a - b))
    assert all(view.inside(p) for p in anim.flow.reshape(-1, 2))


# ---------------------------------------------------------------------- 100
def test_es_replay_is_the_studys_run():
    needs("evolution-strategies")
    s, d = study("evolution-strategies"), M.es_trace()
    assert np.allclose(d["paths"].reshape(-1, 2), s.panels[0].data["xy"])
    assert np.allclose(d["loss"], s.panels[1].data["ys"])
    assert np.allclose(d["alignment"], s.panels[2].data["ys"])
    assert np.allclose(d["network_loss"], s.panels[3].data["ys"])
    assert s.panels[3].notes[0].endswith("%.2f" % d["error"][-1])
    # the first step on the bowl: twin losses, the estimate, and the step it leads to
    w = d["paths"][1, 0]
    for sign, losses in ((1, d["fp0"]), (-1, d["fm0"])):
        p = w + sign * M.ES_SIGMA * d["eps0"]
        assert np.allclose(losses, 0.5 * np.einsum("ni,ij,nj->n", p, M.ES_H, p))
    assert np.allclose(d["true0"], M.ES_H @ w)
    assert np.allclose(d["paths"][1, 1], w - 0.12 * d["est0"])
    # the network: the kept gradient is backprop's, and the first ES update is the kept vote
    x, y, _, _ = S._digit_data(48)
    P = S.mlp_init(49, 5, 10, np.random.default_rng(0))
    loss, grads = S.mlp_loss(S._unpack(d["theta"], P), x, y)
    assert np.isclose(loss, d["network_loss"][0, 0]) and np.allclose(S._pack(grads), d["net_true"])
    after = S.mlp_loss(S._unpack(d["theta"] - 0.15 * d["net_est"], P), x, y, False)[0]
    assert np.isclose(after, d["network_loss"][1, 1])


def test_es_numbers_on_screen():
    needs("evolution-strategies")
    s = study("evolution-strategies")
    anim = M.EvolutionStrategies(s)
    sub = subtitles(anim)
    bowl, align, net = (s.panels[k].data["ys"] for k in (1, 2, 3))
    assert "cosine %.2f" % align[1, 0] in sub[1]
    assert np.isclose(anim.running_cos[-1], align[1, 0])          # the arrow drawn is that one
    assert np.allclose(anim.running[-1], anim.d["est0"])
    assert "%.5f and evolution strategies at %.5f" % (bowl[0, -1], bowl[1, -1]) in sub[2]
    assert "by %.2f" % float(s.panels[3].notes[0].split()[-1]) in sub[3]
    assert "%.2f with backprop and %.2f with" % (net[0, -1], net[1, -1]) in sub[4]
    assert np.allclose(anim.net_running[-1], anim.d["net_est"])
    assert anim.d["theta"].size == 310 and "310 weights" in sub[3]
    # the level lines drawn are level lines of the study's bowl
    ring = M.level_curve(M.ES_H, 3.0)
    assert np.allclose(0.5 * np.einsum("ni,ij,nj->n", ring, M.ES_H, ring), 3.0)

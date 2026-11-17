"""Animations 157-158 (voxanim.mech_anomaly) and their experiments (voxanim.study_anomaly):
the layout is clean, the loop closes, every phase says what kind of evidence it shows, the
numbers on screen are the study's, each study's claim holds as a relative claim with a
margin over seeds, the recorded arrays are what they say, and no threshold or fit has seen
an anomaly.

Both train on MNIST and are skipped without it; the helpers (scores, the mixture, the flow
and its gradient, the combinations) are tested on seeded draws and need nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import data_marks, needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_anomaly as M
from voxanim import study_anomaly as S
from voxanim import study_phys_adv as P
from voxanim.next_catalog import MNIST, build_study

TWO, FLOW = "anomaly-loss-and-latent", "anomaly-two-score-flow"
NAMES = (TWO, FLOW)
EACH = [pytest.param(n, marks=data_marks(n)) for n in NAMES]
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_anomaly", name))
    return _built[name]


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, k, at=0.9):
    """Every string on screen ``at`` of the way through phase ``k``."""
    a, b = anim.spans[k]
    return [x.s for x in anim.scene(min(a + (b - a) * at, b - anim.FADE - 0.05)).texts]


def has(strings, part):
    return any(part in s for s in strings)


def pct(v, digits=0):
    return "%.*f%%" % (digits, 100.0 * float(v))


# ---------------------------------------------------------------------- both
def test_registered_with_what_they_need():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)
    assert set(NAMES) <= MNIST


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    anim, _ = built(name)
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for a, b in anim.spans:                           # every phase draws things, not only text
        assert len(anim.scene((a + b) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    anim, _ = built(name)
    for sub in subtitles(anim):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2 and sub
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(anim.PHASES)):                 # what kind of evidence, in every phase
        for at in (0.3, 0.9):
            assert any(t.startswith(M.TAG_STARTS) for t in texts(anim, k, at))


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    anim, s = built(name)
    m = s.metadata
    assert len(s.panels) == 4 and s.validate() is s
    assert len(m["designs_tried"]) >= 2 and m["limitations"] and m["evidence"]
    assert "model-independent search" in m["counterpart"]
    assert "No anomaly" in m["normal_only"] and m["chosen_by_looking_at_anomalies"]
    assert len(m["seeds"]) >= 3
    # the stand-in and its counterpart, in plain words, on screen in the first phase
    first = texts(anim, 0)
    assert has(first, "stand in for events") and has(first, "model-independent search")


# ---------------------------------------------------------------------- helpers
def test_groups_and_cuts_on_known_inputs():
    yt = np.array([0, 1, 2, 1, 3, 4, 1, 5])
    cal, test, anom = S.groups(yt, 1)
    assert list(cal) == [0, 2, 4] and list(test) == [5, 7] and list(anom) == [1, 3, 6]
    assert S.cut(np.arange(101.0), 0.05) == 95.0
    assert np.allclose(S.quadrants([1, 1, 0, 0], [1, 0, 1, 0]), 0.25)
    assert np.allclose(S.quadrants([1, 1, 1, 0], [0, 0, 0, 0]), [0, 0.75, 0, 0.25])
    assert np.allclose(S.ranks([1.0, 2.0, 3.0, 4.0], [0.0, 2.5, 9.0]), [0.0, 0.5, 1.0])
    assert S.kind_slice(0) == slice(S.HELD, S.HELD + S.N_COR)


def test_spoiling_does_what_its_name_says():
    rng = np.random.default_rng(0)
    v = rng.random((6, 784)).astype(np.float32)
    out = dict(zip(S.KINDS, S.spoil(v, np.random.default_rng(1))))
    assert set(S.KIND_NAMES) == set(S.KINDS) and len(S.KINDS) == 8
    im = v.reshape(6, 28, 28)
    assert np.allclose(out["inverted"], 1 - v) and np.allclose(out["dimmed"], S.DIMMED * v)
    assert np.allclose(out["rot90"].reshape(6, 28, 28)[:, 27, 0], im[:, 0, 0])
    assert np.allclose(out["flip"].reshape(6, 28, 28), im[:, :, ::-1])
    assert np.allclose(out["shift"].reshape(6, 28, 28)[:, :, S.SHIFT:], im[:, :, :-S.SHIFT])
    half = out["half"].reshape(6, 28, 28)
    assert (half[:, 14:] == 0).all() and np.allclose(half[:, :14], im[:, :14])
    assert np.allclose(out["blend"][1], (v[1] + v[0]) / 2)
    assert 0.15 < np.abs(out["noise"] - v).mean() < 0.3 and out["noise"].min() >= 0


def test_latent_scores_on_known_inputs():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4000, 3)) @ np.array([[2.0, 0, 0], [1.0, 1.0, 0], [0, 0, 0.5]]) + 5.0
    mean, L = S.whitener(z)
    u = S.whiten(z, mean, L)
    assert np.allclose(u.mean(0), 0, atol=1e-9) and np.allclose(np.cov(u.T), np.eye(3), atol=1e-3)
    assert np.isclose(S.gauss_nll(np.zeros((1, 3)))[0], 1.5 * np.log(2 * np.pi))
    assert np.isclose(S.gauss_nll(np.array([[3.0, 4.0]]))[0], 12.5 + np.log(2 * np.pi))
    assert np.allclose(S.mahalanobis(S.gauss_nll(u) + 0.7, 0.7, 3), np.sqrt((u ** 2).sum(1)))
    q, ref = rng.normal(size=(50, 4)), rng.normal(size=(300, 4))
    brute = np.sort(np.sqrt(((q[:, None] - ref[None]) ** 2).sum(2)), 1)[:, S.KNN - 1]
    assert np.allclose(S.knn_distance(q, ref), brute, atol=1e-4)


def test_mixture_finds_two_clumps_a_gaussian_cannot():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(-3, 0.5, (1500, 2)), rng.normal(3, 0.5, (1500, 2))])
    held = np.concatenate([rng.normal(-3, 0.5, (500, 2)), rng.normal(3, 0.5, (500, 2))])
    mean, L = S.whitener(x)
    g = S.mix_fit(S.whiten(x, mean, L), np.random.default_rng(1), k=2)
    centres = mean + g["mu"] @ L.T
    assert np.allclose(np.sort(centres[:, 0]), [-3, 3], atol=0.1)
    u = S.whiten(held, mean, L)
    assert S.mix_nll(u, g).mean() < S.gauss_nll(u).mean() - 1.0
    between = S.whiten(np.zeros((1, 2)), mean, L)     # the empty middle: where the mean is
    assert S.gauss_nll(between)[0] < S.gauss_nll(u).mean()        # one Gaussian: "typical"
    assert S.mix_nll(between, g)[0] > S.mix_nll(u, g).mean() + 5  # the mixture: far from both


@pytest.mark.parametrize("d", [2, 8])
def test_flow_gradient_and_fit(d):
    rng = np.random.default_rng(d)
    p = S.flow_init(d, rng)
    x = rng.normal(size=(12, d))
    assert np.abs(S.flow_nll(p, x) - S.gauss_nll(x)).max() < 1.5      # it starts near the Gaussian
    for k in p:                                       # away from that start, for the check
        if k[0] in "Vc":
            p[k] = p[k] + rng.normal(0, 0.3, p[k].shape)
    loss, g = S.flow_loss_grads(p, x)
    assert np.isclose(loss, S.flow_nll(p, x).mean())
    for k in ("W1", "V2", "c0", "b3", "W%d" % (S.FLOW_LAYERS - 1)):
        i = tuple(int(rng.integers(0, n)) for n in p[k].shape)
        q = {a: b.copy() for a, b in p.items()}
        q[k][i] += 1e-5
        assert abs((S.flow_loss_grads(q, x)[0] - loss) / 1e-5 - g[k][i]) < 1e-3 * (1 + abs(g[k][i]))
    for n in range(S.FLOW_LAYERS):                    # every number is changed by some layer
        assert sorted(np.concatenate([p["ia%d" % n], p["ib%d" % n]])) == list(range(d))
    assert {int(i) for n in range(S.FLOW_LAYERS) for i in p["ib%d" % n]} == set(range(d))


def test_flow_fits_what_a_gaussian_cannot():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4000, 1))
    x = np.column_stack([a, a ** 2 + 0.3 * rng.normal(size=(4000, 1))])      # a bent cloud
    mean, L = S.whitener(x)
    u = S.whiten(x, mean, L)
    flow, curve = S.flow_fit(u[:3000], np.random.default_rng(1))
    assert len(curve) == S.FLOW_STEPS // S.FLOW_EVERY and curve[-1] < curve[0] - 0.3
    assert S.flow_nll(flow, u[3000:]).mean() < S.gauss_nll(u[3000:]).mean() - 0.3


def test_combinations_on_known_inputs():
    rng = np.random.default_rng(11)
    n = 6000
    loss, latent = np.exp(rng.normal(-3, 0.5, n)), rng.normal(20, 2, n)
    cal = np.arange(0, n, 2)
    c, cov = S.combine(loss, latent, cal)
    assert set(c) == set(S.METHODS)
    for k in ("loss", "latent"):
        assert abs(c[k][cal].mean()) < 1e-9 and np.isclose(c[k][cal].std(), 1.0)
    assert np.allclose(c["sum"], c["loss"] + c["latent"]) and abs(cov[0, 1]) < 0.05
    # two independent 5% cuts flag 1 - 0.95^2 of normal data; the fair shared rate is half
    a = c["loss"] > S.cut(c["loss"][cal], 0.05)
    b = c["latent"] > S.cut(c["latent"][cal], 0.05)
    assert abs((a | b).mean() - 0.0975) < 0.012
    fair = 1 - S.cut(c["either"][cal], 0.05)
    assert abs(fair - (1 - np.sqrt(0.95))) < 0.006
    assert abs((c["either"] > S.cut(c["either"][cal], 0.05))[1::2].mean() - 0.05) < 0.012
    # the ellipse of two independent standard scores: a chi-square with 2 degrees of freedom
    assert abs(S.cut(c["plane"][cal], 0.05) - (-2 * np.log(0.05))) < 0.5
    # it is two-sided: a digit far *below* normal in both scores is flagged by it alone
    far = S.combine(np.append(loss, np.exp(-6.0)), np.append(latent, 10.0), cal)[0]
    assert far["plane"][-1] > S.cut(far["plane"][cal], 0.05)
    assert far["sum"][-1] < 0 and far["either"][-1] < 0.01


# ---------------------------------------------------------------------- the runs
@needs_mnist
def test_same_networks_as_the_earlier_animation():
    """Retrained from the same random streams: the reconstruction errors are that study's."""
    old = P.ae_arrays()["err"]
    yt = P.ae_data()[3]
    for code in S.CODES:
        d = S.arrays(code)
        assert d["scores"].shape == (10, len(S.SEEDS), len(S.SCORES),
                                     S.HELD + len(S.KINDS) * S.N_COR)
        assert np.allclose(d["scores"][:, :, 0, :S.HELD], old[P.AE_CODES.index(code)],
                           atol=1e-6)
        assert np.isfinite(d["scores"]).all()
    earlier = P.ae_summary()["auc"][P.AE_CODES.index(S.MAIN)]
    assert np.allclose(S.summary(S.MAIN)["auc"][:, :, 0], earlier, atol=1e-6)
    assert S.SEEDS == P.AE_SEEDS and S.HELD == len(yt) == P.AE_TEST


@needs_mnist
def test_recorded_runs_are_what_they_say():
    xt, yt = P.ae_data()[2:4]
    left, k0 = S.FEATURED, 0
    cal, test, anom = S.groups(yt, left)
    assert not set(cal) & set(test) and not set(anom) & set(cal) | set(anom) & set(test)
    assert (yt[cal] != left).all() and (yt[test] != left).all() and (yt[anom] == left).all()
    for code in (S.DRAWN, S.MAIN):
        d = S.arrays(code)
        sc = d["scores"][left, k0].astype(float)
        typ = d["typical"][left, k0]
        assert typ[0] in test and typ[1] in anom
        assert np.isclose(sc[0, typ[1]], np.median(sc[0, anom]), atol=1e-3)
        # the rebuilt pictures kept are the ones whose error was recorded
        assert np.isclose(P.rebuild_error(d["pics"][left, k0, 1], xt[typ[1]]), sc[0, typ[1]],
                          atol=1e-6)
        spoiled = S.spoiled_inputs(left, code, S.SEEDS[k0])
        assert np.allclose(spoiled[S.KINDS.index("inverted")], 1 - xt[test[:S.N_COR]])
        for j in range(len(S.KINDS)):
            at = int(d["cor_typical"][left, k0, j])
            assert np.isclose(P.rebuild_error(d["pics"][left, k0, 2 + j], spoiled[j, at]),
                              sc[0, S.kind_slice(j)][at], atol=1e-6)
        # the Gaussian's score is a Mahalanobis distance from the training codes' mean
        maha = S.mahalanobis(sc[S.GAUSS], float(d["volume"][left, k0]), code)
        assert maha.min() >= 0 and 0.5 * code < (maha[cal] ** 2).mean() < 1.5 * code
    d2 = S.arrays(S.DRAWN)
    z = d2["z"][left, k0].astype(float)
    u = S.whiten(z, d2["mean"][left, k0], d2["L"][left, k0])
    sc = d2["scores"][left, k0].astype(float)
    assert np.allclose(S.gauss_nll(u) + float(d2["volume"][left, k0]), sc[S.GAUSS, :S.HELD],
                       atol=1e-3)
    assert d2["dens"].shape[2:] == (3, S.GRID, S.GRID) and d2["cloud"].shape[2:] == (S.N_CLOUD, 2)
    # the three drawn densities are densities: each sums to about one over the grid
    box = d2["box"][left, k0]
    cell = (box[1] - box[0]) * (box[3] - box[2]) / (S.GRID - 1) ** 2
    for k in range(3):
        assert 0.85 < np.exp(-d2["dens"][left, k0, k].astype(float)).sum() * cell < 1.1


# ---------------------------------------------------------------------- 157
@needs_mnist
def test_two_spaces_data_is_what_it_says():
    _, s = built(TWO)
    m = s.metadata
    p0, p1, p2, p3 = (p.data for p in s.panels)
    xt, yt = P.ae_data()[2:4]
    left = m["featured"]
    cal, test, anom = S.groups(yt, left)
    assert left == 1 and m["codes"] == [2, 8] and m["alpha"] == 0.05
    assert yt[int(p0["ids"][0])] != left and yt[int(p0["ids"][1])] == left
    assert np.allclose(p0["values"][2].ravel(), xt[int(p0["ids"][1])])
    for k in range(3):                                # the error on screen is the pictures'
        got = P.rebuild_error(p0["values"][2 * k + 1].ravel(), p0["values"][2 * k].ravel())
        assert abs(got - p0["errors"][k]) < 1e-5
    cor = int(S.arrays(S.DRAWN)["cor_typical"][left, 0, S.KINDS.index("inverted")])
    assert np.allclose(p0["values"][4].ravel(), 1 - xt[test[cor]])    # a kept digit, inverted
    assert p0["values"][4].mean() > 0.6               # an inverted digit is mostly ink
    sc = S.arrays(S.DRAWN)["scores"][left, 0].astype(float)
    assert np.isclose(p1["thresholds"][0], np.quantile(sc[0, cal], 0.95))
    assert np.isclose(p1["thresholds"][1], np.quantile(sc[2, cal], 0.95))
    assert np.isclose(p1["flagged"][0, 1], (sc[0, anom] > p1["thresholds"][0]).mean())
    assert np.allclose(np.asarray(p1["loss_hist"]).sum(1), 1) and len(p1["unseen"]) == len(anom)
    assert set(np.unique(p1["classes"])) == set(range(10)) - {left}
    # the four cases sum to one, for every kind, class and seed
    assert np.allclose(np.asarray(p2["values"]).sum(1), 1)
    assert np.allclose(np.asarray(p2["by_class"]).sum(2), 1)
    assert p2["labels"][0] == "unseen class" and len(p2["labels"]) == 1 + len(S.KINDS)
    thr = np.asarray(p2["thresholds"])
    sc8 = S.arrays(S.MAIN)["scores"][left, 0].astype(float)
    assert np.isclose(thr[0], np.quantile(sc8[0, cal], 0.95))
    for q, (a, b) in enumerate(np.asarray(p2["example_scores"])):   # each example is its case
        assert (a > thr[0]) == (q in (0, 1)) and (b > thr[1]) == (q in (0, 2))
    a, b = p2["own_scores"]
    assert p2["own_found"] and a < thr[0] and b > thr[1]
    assert np.isclose(p2["run"][0, 2], ((sc8[0, anom] <= thr[0]) & (sc8[2, anom] > thr[1])).mean())
    assert np.allclose(np.asarray(p2["kind_pictures"])[2], 1 - np.asarray(p2["kind_pictures"])[0])
    assert np.asarray(p3["seeds"]).shape == (3, 10, len(S.SEEDS))
    assert np.isclose(p3["ys"][0, left], m["auc_by_class"][str(left)]["loss"]["mean"])
    assert np.isclose(p2["run_auc"][1], P.auc(sc8[2, np.concatenate([cal, test])], sc8[2, anom]))


@needs_mnist
def test_two_spaces_claims():
    _, s = built(TWO)
    m = s.metadata
    _p0, _p1, p2, p3 = (p.data for p in s.panels)
    loss, knn, gauss = np.asarray(p3["seeds"])                # [class, seed] each
    left = m["featured"]
    # the motivating failure, reproduced: the loss ranks unseen 1s below kept digits
    assert loss[left].max() < 0.4
    # their codes do sit apart: the neighbour distance is well above chance, every seed
    assert knn[left].min() > loss[left].max() + 0.3 and knn[left].min() > 0.65
    assert m["rescued_classes"] == [left]
    # but at 5% false alarms most of them are still caught by neither cut
    by_class = np.asarray(p2["by_class"])                     # [class, seed, case]
    assert by_class[left, :, 3].min() > 0.7 and by_class[left, :, 2].min() > 0.1
    assert by_class[left, :, :2].sum(1).max() < 0.02
    # on average the loss is still the better single score, by more than seeds differ
    assert loss.mean(0).min() > knn.mean(0).max() + 0.05
    # one Gaussian (Mahalanobis) is worse than a coin toss; the neighbour distance is not
    assert gauss.mean(0).max() < 0.35 and knn.mean(0).min() > gauss.mean(0).max() + 0.3
    # in a code of two numbers the latent space says nothing
    drawn = np.asarray(p3["drawn"])
    assert drawn[1].mean(0).max() < 0.45 and drawn[2].mean(0).max() < 0.45
    # kinds, by seed [kind, seed, case]: 0 unseen class, then KINDS
    seeds = np.asarray(p2["seeds"])
    kind = {k: seeds[1 + j] for j, k in enumerate(S.KINDS)}
    assert kind["inverted"][:, 0].min() > 0.85                # both
    assert kind["noise"][:, 1].min() > 0.9                    # the loss only
    assert kind["shift"][:, 1].min() > 0.75 and kind["rot90"][:, 1].min() > 0.6
    assert kind["dimmed"][:, 3].min() > 0.9 and kind["blend"][:, 3].min() > 0.9   # neither
    # half a digit erased: the one kind that often shows in the code alone
    rest = np.array([kind[k][:, 2] for k in S.KINDS if k != "half"])
    assert kind["half"][:, 2].min() > rest.max() + 0.05
    # a dimmed digit is rebuilt *better* than a normal one
    kinds = np.asarray(p3["kinds"])                           # [kind, (loss, knn, gauss)]
    assert kinds[S.KINDS.index("dimmed"), 0] < 0.25
    # each cut alone flags about 5% of the other half of the normal digits
    normal = np.asarray(p2["normal"])
    assert 0.035 < normal[0] + normal[1] < 0.065 and 0.035 < normal[0] + normal[2] < 0.065


@needs_mnist
def test_two_spaces_numbers_on_screen():
    anim, s = built(TWO)
    m, sub = s.metadata, subtitles(anim)
    p0, p1, p2, _p3 = (p.data for p in s.panels)
    left = m["featured"]
    by = m["auc_by_class"][str(left)]
    assert "every %d removed" % left in sub[0]
    assert "%s above the distance threshold" % pct(p1["flagged"][1, 1]) in sub[1]
    assert "area %.2f for the distance" % p2["run_auc"][1] in sub[2]
    assert "caught by both (%s)" % pct(m["quadrants"]["inverted"]["both"]) in sub[3]
    assert "most unseen %ds (%s)" % (left, pct(m["quadrants_featured"]["neither"]["mean"])) \
        in sub[3]
    assert "the loss has area %.2f, the distance %.2f" % (
        m["auc_loss"]["mean"], m["auc_knn"]["mean"]) in sub[4]
    assert "it is %.2f against %.2f: the code sees" % (
        by["loss"]["mean"], by["knn"]["mean"]) in sub[4]
    assert "= %.3f" % p0["errors"][2] in texts(anim, 0)
    assert has(texts(anim, 0), ("(%.2f, %.2f)" % tuple(p0["codes"][2])).replace("-", "−"))
    assert has(texts(anim, 1), "%s of unseen %ds" % (pct(p1["flagged"][0, 1]), left))
    assert has(texts(anim, 2), "area %.2f" % p2["run_auc"][0])
    four = texts(anim, 3)
    assert has(four, "loss above %.3f?" % p2["thresholds"][0]) and has(four, "unseen %d" % left)
    assert has(four, "distance %.2f > cut" % p2["own_scores"][1])
    assert has(texts(anim, 4), "the %ds: %.2f, %.2f, %.2f" % (
        left, by["loss"]["mean"], by["knn"]["mean"], by["gauss"]["mean"]))
    assert has(texts(anim, 4), "picked over Mahalanobis")     # the choice made on anomalies
    last = texts(anim, 5)
    assert has(last, "neither: %s" % pct(m["quadrants"]["dimmed to 40%"]["neither"]))
    assert all(has(last, S.KIND_NAMES[k]) for k in S.KINDS)


# ---------------------------------------------------------------------- 158
@needs_mnist
def test_procedure_data_is_what_it_says():
    _, s = built(FLOW)
    m = s.metadata
    p0, p1, p2, p3 = (p.data for p in s.panels)
    yt = P.ae_data()[3]
    left = m["featured"]
    cal, test, anom = S.groups(yt, left)
    lat = (S.GAUSS, S.MIXT, S.FLOW)[int(p1["chosen"])]
    sc = S.arrays(S.MAIN)["scores"][left, 0].astype(float)
    assert m["codes"] == list(S.CODES) and m["alphas"] == [0.01, 0.05]
    assert list(p0["counts"][1:]) == [len(cal), len(test), len(anom)]
    # thresholds are quantiles of calibration digits of kept classes, and of nothing else
    assert np.isclose(p0["cuts"][0], np.quantile(sc[0, cal], 0.95))
    assert np.isclose(p0["cuts"][1], np.quantile(sc[lat, cal], 0.95))
    assert np.isclose(P.rebuild_error(p0["values"][1].ravel(), p0["values"][0].ravel()),
                      p0["event"][0], atol=1e-6)
    assert list(p0["decisions"][:2]) == [p0["event"][0] > p0["cuts"][0],
                                         p0["event"][1] > p0["cuts"][1]]
    # the density is the one that fits calibration digits best: no anomaly in that choice
    nll = np.asarray(p1["normal_nll"])
    assert nll.shape == (10, len(S.SEEDS), 3) and int(p1["chosen"]) == nll.mean((0, 1)).argmin()
    assert np.isclose(nll[left, 0, 2], sc[S.FLOW, cal].mean())
    assert m["density_used"] == S.DENSITIES[int(p1["chosen"])]
    # the plane: standardised on calibration digits; cuts recomputed from them
    c, cov = S.combine(sc[0, :S.HELD], sc[lat, :S.HELD], cal)
    assert np.allclose(p2["anomalies"], np.column_stack([c["loss"], c["latent"]])[anom])
    cuts = np.asarray(p2["cuts"])
    assert np.isclose(cuts[4], np.quantile(c["sum"][cal], 0.95)) and np.allclose(p2["cov"], cov)
    assert cuts[2] > cuts[0] and cuts[3] > cuts[1]            # OR-ed cuts sit further out
    assert np.isclose(p2["caught"][2], ((c["loss"][anom] > cuts[2])
                                        | (c["latent"][anom] > cuts[3])).mean(), atol=0.01)
    assert np.asarray(p3["seeds"]).shape == (len(S.CODES), 10, len(S.SEEDS), len(S.METHODS))
    assert np.allclose(p3["ys"][:, S.CODES.index(S.MAIN)], np.asarray(p3["auc"]).mean((0, 1)))
    # the loss alone is the earlier animation's score, and its area the earlier area
    earlier = P.ae_summary()["auc"]
    for i, code in enumerate(S.CODES):
        assert np.allclose(np.asarray(p3["seeds"])[i, :, :, 0], earlier[P.AE_CODES.index(code)],
                           atol=1e-6)


@needs_mnist
def test_procedure_claims():
    _, s = built(FLOW)
    m = s.metadata
    _p0, p1, _p2, p3 = (p.data for p in s.panels)
    left = m["featured"]
    nll = np.asarray(p1["normal_nll"]).mean(0)                # [seed, density]
    # a mixture and a flow fit kept digits' codes far better than one Gaussian, every seed
    assert nll[:, 0].min() > nll[:, 1:].max() + 1.5
    # the mixture and the flow are close: the choice between them is not a strong one
    assert abs(nll[:, 1].mean() - nll[:, 2].mean()) < 0.3
    area = np.asarray(p1["auc"]).mean(0)                      # [seed, (gauss, mix, flow, knn)]
    assert area[:, 0].max() < 0.35 and area[:, 1:3].min() > 0.5
    assert area[:, 3].min() > area[:, :3].max() + 0.03        # no fitted density beats neighbours
    auc = np.asarray(p3["auc"]).mean(0)                       # [seed, method]
    caught = np.asarray(p3["caught"])                         # [alpha, class, seed, method]
    alarms = np.asarray(p3["alarms"])
    lo, la, ei, su, pl = range(5)
    # on average a second score buys nothing here: either cut is level with the loss in area
    # and catches fewer at a fixed false-alarm rate; the sum and the ellipse are worse
    assert abs(auc[:, ei].mean() - auc[:, lo].mean()) < 0.03
    for a in (0, 1):
        by_seed = caught[a].mean(0)
        assert (by_seed[:, ei] < by_seed[:, lo]).all() and (by_seed[:, su] < by_seed[:, ei]).all()
        assert (by_seed[:, pl] < by_seed[:, su]).all()
    assert auc[:, pl].max() < auc[:, su].min() - 0.05
    # what it buys is the class the loss cannot see at all
    ones = caught[1, left]                                    # [seed, method]
    assert ones[:, lo].max() < 0.01 and ones[:, ei].mean() > 0.05
    assert (ones[:, ei] > ones[:, lo] + 0.01).all()
    one = np.asarray(p3["auc"])[left]                         # [seed, method]: seeds differ here
    assert (one[:, ei] > one[:, lo]).all() and one[:, ei].mean() > one[:, lo].max() + 0.15
    # every threshold delivers about its stated rate on the other half of the normal digits
    assert np.abs(alarms[1].mean((0, 1)) - 0.05).max() < 0.012
    assert np.abs(alarms[0].mean((0, 1)) - 0.01).max() < 0.006
    # the cost of two cuts: about twice one cut's false alarms unless corrected
    raw = np.asarray(p3["or_raw"])                            # [alpha, class, seed, 3]
    assert raw[1, :, :, 1].mean(0).min() > 0.085 and raw[1, :, :, 1].mean(0).max() < 0.125
    assert 0.02 < raw[1, :, :, 2].mean() < 0.032
    assert raw[1, :, :, 0].mean() > caught[1, :, :, ei].mean() + 0.08   # and catches more: unfair
    # code size [code, class, seed, method]: the loss falls, the code's density takes over
    by = np.asarray(p3["seeds"]).mean(1)                      # [code, seed, method]
    i2, i8, i32, i128 = (S.CODES.index(c) for c in (2, 8, 32, 128))
    assert by[i8, :, lo].min() > by[i128, :, lo].max() + 0.15
    assert by[i32, :, la].min() > by[i8, :, la].max() + 0.2
    assert by[i32, :, la].min() > by[i32, :, lo].max() + 0.1
    assert by[i128, :, la].min() > by[i128, :, lo].max() + 0.05
    assert by[i2, :, la].max() < 0.4                          # and nothing in two numbers
    knn = np.asarray(p3["knn_seeds"]).mean(1)                 # [code, seed]
    assert knn[i128].min() > by[i128, :, lo].max() + 0.15
    # the 1s stay hard at a large code, in the code too
    assert np.asarray(p3["seeds"])[i128, left, :, la].max() < 0.5


@needs_mnist
def test_procedure_numbers_on_screen():
    anim, s = built(FLOW)
    m, sub = s.metadata, subtitles(anim)
    p0, p1, p2, p3 = (p.data for p in s.panels)
    assert "(loss %.3f, −ln p %.1f)" % (p0["event"][0], p0["event"][1]) in sub[0]
    nll = np.asarray(p1["normal_nll"]).mean((0, 1))
    assert "%.1f under one Gaussian, %.1f under a mixture, %.1f under a flow" % tuple(nll) in sub[1]
    assert "they catch %s, %s and %s" % tuple(pct(v) for v in p2["caught"][2:]) in sub[2]
    c5, raw = m["caught"]["0.05"], m["or_uncorrected"]["0.05"]
    assert "the loss alone catches %s of anomalies, either cut %s" % (
        pct(c5["loss"]["mean"]), pct(c5["either"]["mean"])) in sub[3]
    assert "flag %s of normal digits, so each is set at %s" % (
        pct(raw["false_alarms"], 1), pct(raw["fair_rate"], 1)) in sub[3]
    by = m["auc_by_code"]
    assert "from %.2f at 8 numbers to %.2f at 128" % (by["8"]["loss"], by["128"]["loss"]) in sub[4]
    assert "from %.2f to %.2f at 32" % (by["8"]["latent"], by["32"]["latent"]) in sub[4]
    one = texts(anim, 0)
    assert has(one, "A  loss = %.3f" % p0["event"][0])
    assert has(one, "threshold %.3f" % p0["cuts"][0]) and has(one, "held-out normal digits")
    two = texts(anim, 1)
    assert has(two, "−ln p of normal codes: %.2f" % nll[2]) and has(two, "no anomaly was looked at")
    assert has(texts(anim, 2), "caught: either cut %s, sum %s, ellipse %s"
               % tuple(pct(v) for v in p2["other_caught"][2:]))
    four = texts(anim, 3)
    assert has(four, "%s of normal digits" % pct(raw["false_alarms"], 1))
    assert has(four, "= %s if independent" % pct(0.0975, 2))
    assert has(four, pct(c5["sum"]["mean"])) and has(four, "%.2f" % m["auc"]["either"]["mean"])
    knn = np.asarray(p3["knn"])
    assert has(texts(anim, 4), "neighbour distance  " + ", ".join("%.2f" % v for v in knn[1:]))

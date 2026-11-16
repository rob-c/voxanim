"""Animations 101-105, four generative families compared: what they draw is what was
measured, on one shared set of models."""

import os

import numpy as np
import pytest
from conftest import available

from voxanim import layout, presets, qa
from voxanim import mech_families as M
from voxanim import study_families as F
from voxanim.next_catalog import build_study

NAMES = ("generative-objectives", "generative-latent-spaces", "generative-sampling",
         "generative-reconstruction", "generative-interpolation")
WIDE = F.CODES.index(F.WIDE)
TWO = F.CODES.index(2)


def needs(name):
    return pytest.mark.skipif(not available(name), reason="MNIST not downloaded")


every = pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in NAMES])
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_families", name))
    return _built[name]


def shown(anim, k, u):
    """Every string on screen ``u`` of the way through phase ``k``, with the live subtitle."""
    a, b = anim.spans[k]
    scene = anim.scene(float(a + (b - a) * u))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def has(strings, part):
    return any(part in s for s in strings)


def fam(name):
    return F.FAMILIES.index(name)


# ---------------------------------------------------------------------- all five
@every
def test_studies_say_what_was_run_and_what_was_held_equal(name):
    _anim, study = built(name)
    meta = study.metadata
    assert set(F.STUDIES) == set(NAMES) == set(M.ANIMATIONS)
    assert meta["dataset"] == "MNIST" and len(meta["seeds"]) >= 3 and meta["codes"] == [2, 16]
    assert len(meta["held_equal"]) >= 4 and len(meta["not_equal"]) >= 3
    assert len(meta["designs_tried"]) >= 2 and meta["limitations"] and meta["scope"]
    assert meta["data"]["train_x_sha256"] != meta["data"]["test_x_sha256"]
    assert os.path.exists(meta["artifact"]) and meta["run_settings"] == F.SETTINGS
    assert len(study.panels) == 4 and study.validate() is study


@every
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(name, samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))

    def key(sc):
        return sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)

    assert key(first) == key(last)
    for k in range(len(anim.PHASES)):                    # a subtitle is two lines at most
        lines = layout.wrap(anim.subtitle_for(k), layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2 and anim.subtitle_for(k)
        assert max(layout.text_width(s, layout.SUB_SIZE) for s in lines) <= 1 - 2 * layout.SAFE


# ---------------------------------------------------------------------- the shared set
@needs(NAMES[0])
def test_one_shared_set_same_data_same_networks():
    d = F.data()
    assert d["x"].shape == (F.TRAIN_N, 784) and d["xt"].shape == (F.TEST_N, 784)
    # no held-out digit is a training digit
    assert F.nearest(d["xt"][:300], d["x"])[0].min() > 0.5
    saved = F.models()
    assert str(saved["train_x_sha256"]) == d["train_x_sha256"]
    for code in F.CODES:
        for seed in F.SEEDS:
            ae, dec, idec = (F.params(f, code, seed) for f in ("ae", "dec", "idec"))
            for family in F.FAMILIES:                    # one decoder shape for all
                P = F.params(family, code, seed)
                assert [P["W%d" % k].shape for k in (3, 4, 5)] == [(code, 64), (64, 256),
                                                                    (256, 784)]
            assert "W0" not in F.params("gan", code, seed)          # a GAN has no encoder
            # DEC: the autoencoder's decoder untouched, its encoder moved; IDEC: both moved
            assert all(np.array_equal(dec["W%d" % k], ae["W%d" % k]) for k in (3, 4, 5))
            assert not np.allclose(dec["W2"], ae["W2"])
            assert not np.allclose(idec["W5"], ae["W5"])
            assert dec["mu"].shape == idec["mu"].shape == (F.CLUSTERS, code)
            # both start from that seed's autoencoder: the same rebuilding error at update 0
            assert F.trace("dec", code, seed, "mse")[0] == pytest.approx(
                F.trace("ae", code, seed, "mse")[-1], rel=1e-4)


@needs(NAMES[0])
def test_gan_gradients_and_the_noise_search():
    rng = np.random.default_rng(0)
    D = F.judge_init(rng)
    x = rng.random((3, 784))
    logit, tape = F.judge_forward(D, x)
    grads, dx = F.judge_backward(D, tape, np.ones_like(logit))
    e = 1e-6
    for k, idx in (("Wc", (5, 0)), ("Wa", (300, 7))):
        plus = {a: b.copy() for a, b in D.items()}
        plus[k][idx] += e
        fd = (F.judge_forward(plus, x)[0].sum() - logit.sum()) / e
        assert fd == pytest.approx(grads[k][idx], rel=1e-3, abs=1e-6)
    x2 = x.copy()
    x2[1, 400] += e
    assert (F.judge_forward(D, x2)[0].sum() - logit.sum()) / e == pytest.approx(
        dx[1, 400], rel=1e-3, abs=1e-6)
    # the search finds a digit the generator can draw
    G = F.params("gan", 2, F.SEEDS[0])
    target = F.decode(G, rng.normal(size=(4, 2)))
    _z, err, curve, _zs, _imgs = F.invert(G, target, 2, rng)
    assert err.max() < 0.005 and curve[-1] < curve[0] * 0.05


# ---------------------------------------------------------------------- 101 objectives
@needs("generative-objectives")
def test_objectives_each_family_is_ahead_on_what_it_is_trained_for():
    _anim, study = built("generative-objectives")
    v, defined = F.yardsticks()                          # family, yardstick, seed
    assert np.array_equal(study.panels[2].data["seeds"], v) and v.shape[2] >= 3
    ae, vae, gan, dec, idec = (fam(f) for f in F.FAMILIES)
    # rebuilding: the AE is far ahead of DEC, whose decoder is left behind; IDEC keeps it
    assert v[ae, 0].max() < v[dec, 0].min() - 0.03
    assert v[idec, 0].max() < v[dec, 0].min() - 0.03
    assert v[ae, 0].max() < v[gan, 0].min() - 0.005      # and ahead of a search of the GAN
    # the bell: only the VAE's codes are inside it
    assert v[vae, 1].min() > 0.95 and max(v[k, 1].max() for k in (ae, dec, idec)) < 0.05
    # a reader's confidence in new samples: the GAN ahead of the VAE and the AE, every seed
    assert v[gan, 3].min() > v[vae, 3].max() + 0.05
    assert v[gan, 3].min() > v[ae, 3].max() + 0.1
    # clusters: DEC improves on k-means in the AE's codes it starts from, seed by seed
    assert (v[dec, 2] > v[ae, 2]).all() and v[dec, 2].mean() > v[ae, 2].mean() + 0.02
    assert not defined[gan, 2] and defined.sum() == defined.size - 1
    # DEC's rebuilding error rises through its own run; IDEC's stays near where it started
    tr = F.run_traces()
    assert (tr["mse"]["dec"][:, -1] > 3 * tr["mse"]["dec"][:, 0]).all()
    assert (tr["mse"]["idec"][:, -1] < 2 * tr["mse"]["idec"][:, 0]).all()
    assert (tr["acc"]["dec"][:, -1] > tr["acc"]["dec"][:, 0]).all()


@needs("generative-objectives")
def test_objectives_draws_the_measured_passes_and_table():
    anim, study = built("generative-objectives")
    d = F.data()
    x = d["xt"][F.TRACK]
    ae = F.params("ae", F.WIDE, M.SEED)
    assert np.allclose(anim.ae_code, F.encode(ae, x[None])[0])
    assert anim.err["ae"] == pytest.approx(np.mean((F.decode(ae, anim.ae_code)[0] - x) ** 2))
    # the VAE's noisy codes are its centre plus its own spread times standard noise
    assert np.allclose((anim.vae_z - anim.vae_mu) / anim.vae_sd,
                       np.random.default_rng(3).normal(size=anim.vae_z.shape))
    assert has(shown(anim, 1, 0.9), "%.3f" % anim.err["ae"])
    k = len(anim.gan_steps) - 1
    assert has(shown(anim, 2, 0.97), "right on %s" % M.pct(anim.gan_right[k]))
    assert has(shown(anim, 3, 0.97), "cluster accuracy %.2f" % anim.dec_acc[-1])
    table = np.asarray(study.panels[2].data["values"])
    strings = shown(anim, 5, 0.95)
    assert has(strings, "%.3f" % table[fam("ae"), 0])
    assert has(strings, "%.2f" % table[fam("gan"), 3])
    assert has(strings, "no clusters") and has(strings, "by design")
    assert has(strings, "%.2f to %.2f" % (table[fam("ae"), 2], table[fam("dec"), 2]))
    # the runs' pictures are the saved snapshots of the tracked digit
    assert np.allclose(anim.shots[fam("ae"), -1],
                       F.trace("ae", F.WIDE, M.SEED, "shown")[-1, F.TRACK])


# ---------------------------------------------------------------------- 102 latent spaces
@needs("generative-latent-spaces")
def test_latent_spaces_the_bell_neighbours_and_clusters():
    anim, study = built("generative-latent-spaces")
    d = F.latent_arrays()
    ae, vae, dec, idec = range(4)
    inside, used, pure, acc = (np.asarray(d[k])[TWO] for k in ("inside", "used", "purity",
                                                                "accuracy"))
    assert inside.shape == (4, len(F.SEEDS))
    # the VAE's codes are in the bell and the bell is full of them; nobody else's
    assert inside[vae].min() > max(inside[k].max() for k in (ae, dec, idec)) + 0.25
    assert used[vae].min() > used[ae].max() + 0.1 and used[ae].min() > used[dec].max() + 0.1
    # DEC: fewer same-digit neighbours than the AE it started from, seed by seed, and
    # more accurate clusters than k-means on that AE's codes, seed by seed
    assert (pure[dec] < pure[ae] - 0.05).all()
    assert (acc[dec] > acc[ae]).all()
    assert np.allclose(study.panels[2].data["values"], pure.mean(1))
    # codes are the encoders' outputs for the held-out digits
    xt, yt = F.data()["xt"], F.data()["yt"]
    assert np.allclose(anim.codes[0], F.encode(F.params("ae", 2, M.SEED), xt), atol=1e-5)
    assert pure[ae, 0] == pytest.approx(F.purity(anim.codes[0], yt))
    # the GAN's plane: what its generator draws there, read by the reader
    grid = anim.tile_grid
    assert np.allclose(anim.tiles, F.decode(F.params("gan", 2, M.SEED), grid), atol=1e-5)
    assert np.asarray(d["gan_class"]).shape == (len(F.SEEDS), F.MAP_SIDE, F.MAP_SIDE)
    assert d["gan_purity"].min() > 0.6


@needs("generative-latent-spaces")
def test_latent_spaces_shows_its_numbers():
    anim, study = built("generative-latent-spaces")
    m = study.metadata
    strings = shown(anim, 1, 0.95)
    for i, f in enumerate(F.ENCODERS):  # noqa: B007
        assert has(strings, "%s inside the circle" % M.pct(m["inside_bell"][f]["mean"]))
    # the close-up's count is that digit's own neighbours
    z, y = anim.codes[0], anim.labels
    d2 = ((z - z[anim.focus]) ** 2).sum(1)
    d2[anim.focus] = np.inf
    near = np.argsort(d2)[:10]
    assert set(near) == set(anim.near) and (y[near] == y[anim.focus]).sum() == anim.same
    strings = shown(anim, 3, 0.97)
    assert has(strings, "%d of its 10 nearest" % anim.same)
    assert has(strings, "%.2f" % m["purity"]["dec"]["mean"])
    assert has(strings, "%.2f" % m["pixel_purity"])
    strings = shown(anim, 4, 0.97)
    assert has(strings, M.pct(m["bell_used"]["vae"]["mean"])) and has(strings, "by design")
    hit = F.nearest(anim.bell, anim.codes[F.ENCODERS.index("dec")])[0] <= F.REACH
    assert np.array_equal(hit, anim.lands["dec"])
    assert has(strings, "%s of these 500 draws" % M.pct(hit.mean()))


# ---------------------------------------------------------------------- 103 sampling
@needs("generative-sampling")
def test_sampling_what_the_measures_say():
    _anim, study = built("generative-sampling")
    m = np.asarray(study.panels[2].data["seeds"])        # method, seed, measure
    real = np.asarray(study.panels[2].data["real"])
    row = {k: i for i, k in enumerate(F.METHODS)}
    conf, covered, variety, near = (m[:, :, k] for k in range(4))
    assert m.shape == (len(F.METHODS), len(F.SEEDS), 4) and len(F.SEEDS) >= 3
    # an AE's bell draws: fewest digits, least variety, furthest from any training digit
    for other in ("ae_fit", "vae", "gan", "idec"):
        assert covered[row["ae"]].max() < covered[row[other]].min()
        assert near[row["ae"]].min() > near[row[other]].max() + 1.0
    assert variety[row["ae"]].max() < variety[row["gan"]].min() - 1.5
    # the GAN: the reader's confidence and the variety ahead of the VAE's, every seed
    assert conf[row["gan"]].min() > conf[row["vae"]].max() + 0.05
    assert variety[row["gan"]].min() > variety[row["vae"]].max() + 1.0
    assert variety[row["gan"]].max() < real[2]            # no family is as varied as real digits
    assert conf.max() < real[0]
    # nobody copies: every family's draws are as far from the training set as real digits
    # are, or nearly (a copy would be at 0)
    assert near.min() > 0.85 * real[3]
    assert np.allclose(study.panels[1].data["values"], conf.mean(1))


@needs("generative-sampling")
def test_sampling_draws_are_the_models_own_and_the_numbers_shown():
    anim, study = built("generative-sampling")
    rng = np.random.default_rng([M.SEED, F.WIDE, 7])
    eps = rng.normal(size=(F.SAMPLES, F.WIDE))
    pick = rng.integers(0, F.CLUSTERS, F.SAMPLES)
    assert np.array_equal(anim.eps, eps[:F.KEPT])
    for method in ("ae", "vae", "gan", "idec"):
        img = F.draw(method, F.WIDE, M.SEED, eps[:F.KEPT], pick[:F.KEPT])
        assert np.allclose(anim.kept[F.METHODS.index(method)], img, atol=1e-5)
    # a draw's "nearest training digit" is one, at that distance
    x = F.data()["x"]
    g = F.METHODS.index("gan")
    dist, index = F.nearest(anim.kept[g, :3], x)
    assert np.allclose(anim.near_img[g, :3], x[index], atol=1e-6)
    assert np.allclose(anim.near_dist[g, :3], dist, atol=1e-4)
    t = study.metadata["table"]
    strings = shown(anim, 2, 0.97)
    assert has(strings, "%.2f" % t["gan"]["confidence"]["mean"]) and has(strings, "10 of 10")
    assert has(strings, "%d to %d" % (t["ae"]["covered"]["min"], t["ae"]["covered"]["max"]))
    strings = shown(anim, 3, 0.97)
    assert has(strings, "%.1f" % t["vae"]["variety"]["mean"])
    strings = shown(anim, 4, 0.97)
    assert has(strings, "%.1f" % t["idec"]["nearest"]["mean"])
    assert has(strings, "%.1f" % anim.near_dist[g, 0])


# ---------------------------------------------------------------------- 104 reconstruction
@needs("generative-reconstruction")
def test_reconstruction_order_and_the_search_control():
    _anim, study = built("generative-reconstruction")
    d = F.recon_arrays()
    mse, right, sharp = (np.asarray(d[k])[WIDE] for k in ("mse", "right", "sharp"))
    ae, vae, gan, dec, idec = (fam(f) for f in F.FAMILIES)
    assert mse.shape == (5, len(F.SEEDS))
    # AE and VAE, then IDEC, then the GAN by search, then DEC with a stale decoder
    assert max(mse[ae].max(), mse[vae].max()) < mse[idec].min() - 0.003
    assert mse[idec].max() < mse[gan].min() - 0.005
    assert mse[gan].max() < mse[dec].min() - 0.02
    assert abs(mse[ae].mean() - mse[vae].mean()) < 0.001      # no winner between these two
    # the control: the search returns the generator's own samples almost exactly, so the
    # GAN's error on held-out digits is what its generator cannot draw
    own = np.asarray(d["own_mse"])[WIDE]
    assert own.max() < 0.001 and own.max() < 0.05 * mse[gan].min()
    # what is lost: the GAN keeps strokes sharper than the AE and VAE but loses the digit
    # more often; DEC loses it about half the time
    assert sharp[gan].min() > max(sharp[ae].max(), sharp[vae].max()) + 0.05
    assert right[gan].max() < min(right[ae].min(), right[vae].min()) - 0.03
    assert right[dec].max() < 0.6 < right[idec].min()
    assert np.allclose(study.panels[1].data["values"], mse.mean(1))


@needs("generative-reconstruction")
def test_reconstruction_shows_the_real_search_and_numbers():
    anim, study = built("generative-reconstruction")
    inv = F.inversions()
    z = inv["c%d_s%d_z" % (F.WIDE, M.SEED)][F.TRACK]
    out = F.decode(F.params("gan", F.WIDE, M.SEED), z)[0]
    assert np.allclose(anim.path_img[-1], out, atol=1e-5)          # the search's last image
    assert np.allclose(anim.kept[fam("gan"), F.TRACK], out, atol=1e-5)
    assert np.allclose(anim.path_z[-1], z)
    assert (np.diff(anim.path_err) < 1e-4).all() and anim.path_err[-1] < 0.5 * anim.path_err[0]
    strings = shown(anim, 1, 0.97)
    assert has(strings, "step %d" % F.INV_STEPS) and has(strings, "error %.3f" % anim.path_err[-1])
    m = study.metadata
    assert has(strings, "%.3f" % m["mse"]["gan"]["mean"])
    strings = shown(anim, 3, 0.97)
    for f in F.FAMILIES:
        assert has(strings, "%.3f" % m["mse"][f]["mean"])
        assert has(strings, "%.3f" % m["mse_2"][f]["mean"])
    strings = shown(anim, 4, 0.97)
    assert has(strings, "same digit: %s" % M.pct(m["read_right"]["dec"]["mean"]))
    assert has(strings, "sharpness: %.2f" % m["sharpness"]["gan"]["mean"])


# ---------------------------------------------------------------------- 105 interpolation
@needs("generative-interpolation")
def test_interpolation_the_middle_of_the_walk():
    _anim, study = built("generative-interpolation")
    d = F.interp_arrays()
    w = np.asarray(d["walk"])[WIDE]                      # seed, walker, measure, step
    row = {k: i for i, k in enumerate(F.WALKERS)}
    mid = F.WALK // 2
    # the reader is less sure half-way than at the ends, on every walk and seed
    assert (w[:, :, 0, mid] < 0.5 * (w[:, :, 0, 0] + w[:, :, 0, -1]) - 0.05).all()
    # by the reader's confidence no encoder's walk beats the cross-fade by much
    for f in ("ae", "vae", "idec"):
        assert abs(w[:, row[f], 0, mid].mean() - w[0, row["pixels"], 0, mid]) < 0.05
    # distance to the nearest training digit: the cross-fade's middle strays furthest
    further = F.bump(w[:, :, 1])
    for f in F.FAMILIES:
        assert further[:, row[f]].max() < further[0, row["pixels"]] - 0.2
    assert np.allclose(study.panels[3].data["values"], further.mean(0))
    # DEC's ends are already wrong about half the time: its flat line is not a good walk
    assert study.metadata["end_neither"]["dec"]["mean"] > 0.4
    pairs = np.asarray(d["pairs"])
    y = F.data()["yt"]
    assert len(pairs) == F.WALK_PAIRS and (y[pairs[:, 0]] != y[pairs[:, 1]]).all()


@needs("generative-interpolation")
def test_interpolation_walks_are_straight_lines_between_real_codes():
    anim, study = built("generative-interpolation")
    pairs = F.walk_pairs()
    xt = F.data()["xt"]
    a, b = pairs[0]
    assert np.allclose(anim.ends[0], xt[[a, b]])
    for family in ("ae", "gan"):
        z = F.codes_of(family, F.WIDE, M.SEED)
        P = F.params(family, F.WIDE, M.SEED)
        walk = anim.kept[F.WALKERS.index(family), 0]
        assert np.allclose(walk[0], F.decode(P, z[a])[0], atol=1e-5)
        assert np.allclose(walk[F.WALK // 2], F.decode(P, 0.5 * (z[a] + z[b]))[0], atol=1e-5)
        assert np.allclose(walk[-1], F.decode(P, z[b])[0], atol=1e-5)
    fade = anim.kept[F.WALKERS.index("pixels"), 0]
    assert np.allclose(fade[F.WALK // 2], 0.5 * (xt[a] + xt[b]), atol=1e-6)
    m = study.metadata
    strings = shown(anim, 2, 0.97)
    assert has(strings, "%.2f" % m["middle_confidence"]["gan"]["mean"])
    assert has(strings, M.pct(m["middle_neither"]["pixels"]["mean"]))
    strings = shown(anim, 3, 0.97)
    assert has(strings, "+%.2f" % m["middle_further"]["pixels"]["mean"])
    assert has(strings, "%.1f" % anim.mids["gan"][2])

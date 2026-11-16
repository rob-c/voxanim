"""Animations 59-66 of the "next 100": what they draw is what was measured."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_generative as M
from voxanim import presets, qa
from voxanim import study_generative as sg
from voxanim import toy2d as toy
from voxanim.next_catalog import build_study

NAMES = ("gan-training", "normalizing-flows", "flow-matching", "classifier-free-guidance",
         "ddpm-vs-ddim", "vae-posterior-collapse", "vq-vae-codebook", "autoregressive-pixels")


def needs(name):
    return pytest.mark.skipif(not available(name), reason="MNIST not downloaded")


_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("next_vision", name))
    return _built[name]


def shown(anim, k, u):
    """Every string on screen ``u`` of the way through phase ``k``, with the live subtitle."""
    a, b = anim.spans[k]
    scene = anim.scene(float(a + (b - a) * u))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def has(strings, part):
    return any(part in s for s in strings)


def disjoint(d):
    """The training and held-out images a study recorded are different arrays."""
    return str(d["train_x_sha256"]) != str(d["test_x_sha256"])


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in sg.STUDIES])
def test_redesigned_studies_are_the_catalogues_and_say_where_they_came_from(name):
    _anim, study = built(name)
    meta = study.metadata
    assert meta["redesigned"] and meta["dataset"] == "MNIST" and len(meta["seeds"]) >= 3
    assert len(meta["designs_tried"]) >= 2 and meta["run_settings"]["version"] >= 1
    with np.load(meta["artifact"], allow_pickle=False) as saved:
        d = sg.arrays(name)
        assert set(d) <= set(saved.files)
        assert any("weights_" in key for key in saved.files)
    assert meta["data"]["train_x_sha256"] != meta["data"]["test_x_sha256"]
    assert len(study.panels) == 4 and study.validate() is study


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in NAMES])
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(name, samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))
    key = lambda sc: sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)  # noqa: E731
    assert key(first) == key(last)
    assert all(anim.subtitle_for(k) for k in range(len(anim.PHASES)))


# ---------------------------------------------------------------------- 59 gan-training
@needs("gan-training")
def test_gan_study_plain_gan_drops_digits_more_than_the_crowd_aware_one():
    _anim, study = built("gan-training")
    d = sg.arrays("gan-training")
    covered, all_three, fewest = sg.gan_summary(d)
    assert all_three.shape == (2, len(sg.GAN_SEEDS)) and len(sg.GAN_SEEDS) >= 3
    # the claim: over the second half of training the crowd-aware GAN holds all three
    # digits more of the time, on average by a margin and in every seed against every seed
    assert all_three[1].mean() > all_three[0].mean() + 0.1
    assert all_three[1].min() > all_three[0].max()
    assert fewest[1].min() > fewest[0].min()
    assert np.array_equal(study.panels[3].data["values"], all_three)
    assert study.metadata["designs_tried"] and study.metadata["covered_at_end"]
    # the target: real 0s, 1s and 4s in three clusters a held-out digit falls into
    plane = sg.digit_plane()
    assert set(np.unique(plane["train_labels"])) == set(sg.DIGITS) and disjoint(d)
    index, distance = sg.nearest_real(plane["test"], plane["train"])
    assert np.mean(plane["train_labels"][index] == plane["test_labels"]) > 0.95
    assert np.mean(distance <= float(d["reach"])) == pytest.approx(0.99, abs=0.002)
    # the counts are of the saved samples
    again = sg.gan_shares(d["fakes"][0, 0, -1], plane, float(d["reach"]))
    assert np.allclose(again, d["shares"][0, 0, -1])
    assert np.array_equal(covered, (d["shares"] >= sg.GAN_COVERED).sum(-1))
    for row, kind in enumerate(("plain", "crowd")):
        generator = M.weights(d, kind + "_generator_weights_")
        assert np.allclose(toy.forward(generator, d[kind + "_noise"])[0], d["fakes"][row, 0, -1],
                           atol=1e-5)


@needs("gan-training")
def test_gan_animation_draws_the_studys_runs_and_their_coverage():
    anim, study = built("gan-training")
    d = sg.arrays("gan-training")
    assert anim.fakes is d["fakes"] and np.array_equal(anim.real, d["real"])
    assert np.array_equal(study.panels[1].data["xy"], d["fakes"][0, 0, -1])
    _covered, all_three, fewest = sg.gan_summary(d)
    assert has(shown(anim, 1, 0.5), "only %.0f%% of the checkpoints" % (100 * all_three[0, 0]))
    last = shown(anim, 3, 0.95)
    for kind in range(2):
        assert has(last, "all three at %.0f%%" % (100 * all_three[kind].mean()))
        assert has(last, "fewest at once: %d" % fewest[kind].min())
        for value in all_three[kind]:
            assert "%.0f%%" % (100 * value) in last
    assert has(last, "%.0f%% to %.0f%%" % (100 * all_three[0].min(), 100 * all_three[0].max()))
    # a digit below the coverage share is called dropped while it is
    lost = np.argwhere(d["shares"][0, 0] < sg.GAN_COVERED)
    assert len(lost)
    at = int(lost[len(lost) // 2][0])
    u = 0.04 + 0.86 * at / (len(d["steps"]) - 1)
    assert has(shown(anim, 1, u), "dropped")


# ---------------------------------------------------------------------- 60 normalizing-flows
@needs("normalizing-flows")
def test_flow_study_beats_the_best_single_gaussian_held_out():
    _anim, study = built("normalizing-flows")
    d = sg.arrays("normalizing-flows")
    held = d["nll"][:, 1]
    assert len(held) == len(sg.FLOW_SEEDS) >= 3 and disjoint(d)
    # every seed's flow is better than one Gaussian fitted to the same training points
    assert held.max() < float(d["fitted_gaussian"]) - 0.5
    assert float(d["fitted_gaussian"]) <= float(d["bell"][1])
    assert np.allclose(study.panels[3].data["values"],
                       np.r_[d["bell"][1], d["fitted_gaussian"], held])
    # the Gaussian baseline is the training points' own mean and covariance, held out
    plane = sg.digit_plane()
    assert float(d["fitted_gaussian"]) == pytest.approx(
        sg.gaussian_nll(plane["test"], plane["train"].mean(0), np.cov(plane["train"].T)))
    assert held[0] == pytest.approx(toy.flow_loss_grads(M.weights(d), plane["test"])[0])
    assert float(d["error"]) < 1e-9


@needs("normalizing-flows")
def test_flow_layers_end_at_the_studys_samples_and_likelihood_is_shown():
    anim, study = built("normalizing-flows")
    d = sg.arrays("normalizing-flows")
    assert anim.states.shape == (7, 240, 2)
    assert np.array_equal(anim.states[0], d["z"])
    assert np.array_equal(anim.states[-1], d["final"])
    assert np.array_equal(anim.states[-1], study.panels[1].data["xy"])
    assert np.array_equal(anim.mesh[:, -1], d["grid"])
    # each layer moves one coordinate only, by the shift and stretch it reports
    for k in range(6):
        moved = np.abs(anim.states[k + 1] - anim.states[k]).max(0)
        assert moved.min() == 0.0 < moved.max()
    i, j, shift, stretch = M.coupling_terms(anim.p, anim.layer, anim.z)
    assert np.allclose((anim.z[:, j] - shift) * stretch, anim.states[1][:, j])
    assert np.array_equal(anim.z[:, i], anim.states[1][:, i])
    held = d["nll"][:, 1]
    last = shown(anim, 4, 0.95)
    assert has(last, "%.2f nats per held-out digit (three seeds: %.2f to %.2f) against %.2f" % (
        held.mean(), held.min(), held.max(), float(d["fitted_gaussian"])))
    for value in (*held, float(d["bell"][1]), float(d["fitted_gaussian"])):
        assert "%.2f" % value in last
    # the exact log density shown for one digit is the flow's own
    z, logdet, _ = toy.flow_forward(anim.p, anim.x[anim.one][None])
    assert anim.one_bell == pytest.approx(float(-0.5 * (z ** 2).sum() - np.log(2 * np.pi)))
    assert anim.logdet[anim.one] == pytest.approx(float(logdet[0]))


# ---------------------------------------------------------------------- 61 flow-matching
@needs("flow-matching")
def test_flow_matching_few_step_paths_reproduce_the_studys_errors():
    anim, study = built("flow-matching")
    d = M.raw("flow-matching")
    for m in range(2):
        fine = M.euler_path(anim.models[m], anim.start, 40)
        assert np.allclose(fine, anim.paths[m])
        for k, _steps in enumerate(anim.FEW):
            miss = np.mean((anim.few[m][k][-1] - anim.paths[m][-1]) ** 2)
            assert miss == pytest.approx(d["few_error"][m, k])
    excess = [float(v) for v in study.panels[3].data["values"]]
    assert has(shown(anim, 1, 0.5), "paths %.2f longer" % excess[0])
    assert has(shown(anim, 2, 0.5), "only %.3f longer" % excess[1])
    assert has(shown(anim, 3, 0.95), "%.2g" % d["few_error"][0, 0])
    assert excess[1] < excess[0]


# ---------------------------------------------------------------------- 62 classifier-free-guidance
@needs("classifier-free-guidance")
def test_guidance_study_class_match_rises_with_the_dial():
    _anim, study = built("classifier-free-guidance")
    d = sg.arrays("classifier-free-guidance")
    match, variety = d["match"], d["variety"]
    assert match.shape == (len(sg.CFG_SEEDS), len(sg.CFG_SCALES)) and len(sg.CFG_SEEDS) >= 3
    assert sg.CFG_SCALES == (0, 1, 3, 8) and disjoint(d)
    # ignoring the label is chance; telling it helps; guidance 3 helps much more: every
    # denoiser against every denoiser, with a margin
    assert match[:, 0].max() < 0.25
    assert match[:, 1].min() > match[:, 0].max() + 0.3
    assert match[:, 2].min() > match[:, 1].max() + 0.15
    assert match[:, 3].min() > 0.9
    # variety within a digit falls to guidance 3; at 8 it is back up (thicker strokes)
    assert variety[:, 2].max() < variety[:, 0].min() - 0.005
    assert (variety[:, 3] > variety[:, 2]).all() and (d["ink"][:, 3] > d["ink"][:, 0]).all()
    assert np.allclose(study.panels[3].data["ys"], [match.mean(0), variety.mean(0)])
    # the judge is a separate, competent classifier, and the numbers are of the saved samples
    assert float(d["judge_accuracy"]) > 0.95
    judge = M.weights(d, "judge_weights_")
    labels = sg.cfg_labels()
    for row in range(len(sg.CFG_SCALES)):
        out = d["finals"][row]
        assert out.min() >= 0.0 and out.max() <= 1.0
        read = toy.forward(judge, out.astype(float))[0].argmax(1)
        assert np.mean(read == labels) == pytest.approx(match[0, row])
        assert out.reshape(10, sg.CFG_PER_DIGIT, -1).std(1).mean() == pytest.approx(
            variety[0, row])


@needs("classifier-free-guidance")
def test_guidance_animation_draws_the_studys_sampler_and_scores():
    anim, study = built("classifier-free-guidance")
    d = sg.arrays("classifier-free-guidance")
    P, abar, ts = M.weights(d), d["abar"], d["timesteps"]
    labels = sg.cfg_labels()
    shown_rows = list(range(0, len(labels), sg.CFG_PER_DIGIT))
    assert np.array_equal(d["finals"][:2][:, shown_rows].reshape(-1, 14, 14),
                          study.panels[1].data["values"])
    # the saved chain is the sampler's: the rule drawn in phase 2, e0 + w (ec - e0), then
    # the clipped clean-image estimate, at the step the animation looks at
    k = anim.AT
    for row, w in enumerate(sg.CFG_SCALES):
        x = d["trace_x"][row, k]
        e0, _ec, eps = sg.cfg_guess(P, x, ts[k], np.arange(10), w)
        assert np.allclose(e0, d["trace_e0"][row, k], atol=1e-5)
        assert np.allclose(eps, d["trace_e0"][row, k] + w * (
            d["trace_ec"][row, k] - d["trace_e0"][row, k]), atol=1e-5)
        x0, nxt = sg.cfg_step(x, eps, abar[ts[k]], abar[ts[k + 1]])
        assert np.allclose(x0, d["trace_x0"][row, k], atol=1e-5)
        assert np.allclose(nxt, d["trace_x"][row, k + 1], atol=1e-4)
    # sampling again from the same noise gives the saved pictures
    noise = np.random.default_rng(sg.CFG_NOISE_SEEDS[0]).standard_normal(
        (len(labels), sg.CFG_PIXELS), dtype=np.float32)
    again, _ = sg.cfg_sample(P, abar, labels, noise, 3)
    assert np.allclose(again, d["finals"][2], atol=1e-4)
    # dial 1 is the plain told-the-digit guess, and what phase 2 shows as implied is x0
    one = anim.scales.index(1)
    assert np.allclose(anim.implied(anim.e0 + 1 * anim.gap), d["trace_x0"][one, k, anim.WHICH],
                       atol=1e-5)
    last = shown(anim, 3, 0.95)
    mean = d["variety"].mean(0)
    assert has(last, "variety within a digit falls %.3f to %.3f, then rises at 8" % (
        mean[0], mean[2]))
    lo, hi = d["match"].min(0), d["match"].max(0)
    assert has(last, "%d-%d%% at dial 0" % (round(1000 * lo[0]) // 10, round(1000 * hi[0]) // 10))
    assert has(last, "%d samples a dial" % len(labels))


# ---------------------------------------------------------------------- 63 ddpm-vs-ddim
@needs("ddpm-vs-ddim")
def test_sampler_replays_end_at_the_studys_images_and_scores_are_shown():
    anim, study = built("ddpm-vs-ddim")
    d = M.raw("ddpm-vs-ddim")
    r = anim.r
    assert np.array_equal(r["ddpm"][0], d["initial_noise"])
    assert np.array_equal(M.picture(r["ddpm"][-1]), d["ddpm"])
    for k, n in enumerate(M.DDIM_STEPS):
        chain = r["ddim%d" % n]
        assert len(chain) == n + 1 and np.array_equal(chain[0], d["initial_noise"])
        assert np.array_equal(M.picture(chain[-1]), d["ddim"][k])
    scores = [float(v) for v in study.panels[3].data["values"]]
    last = shown(anim, 4, 0.95)
    assert has(last, "%.2f after 10 DDIM steps" % scores[0])
    for value in (*scores, anim.ddpm_score):
        assert "%.2f" % value in last
    # the pixel followed is inked by both samplers
    assert M.picture(anim.trace[0][-1]) > 0.7 and M.picture(anim.trace[1][-1]) > 0.7


# ---------------------------------------------------------------------- 64 vae-posterior-collapse
@needs("vae-posterior-collapse")
def test_vae_probe_matches_the_studys_outputs_and_final_kl_is_shown():
    anim, study = built("vae-posterior-collapse")
    d = M.raw("vae-posterior-collapse")
    for k in range(2):
        assert np.allclose(anim.probe[k]["out"][:4], d["recon"][k])
        assert np.allclose(anim.kl[k], d["kl"][k].T)
    left = [float(v) for v in study.panels[3].data["values"]]
    assert np.allclose([e.mean() for e in anim.left_each], left)
    last = shown(anim, 3, 0.95)
    assert has(last, "%.3f nats per number under strong pressure, %.3f with the ramp" % (
        left[0], left[1]))
    # the code is ignored after strong pressure, and not after the ramp
    assert anim.swap[0] < 1e-4 < anim.swap[1]
    # the pressure drawn is the study's schedule: 4 throughout, or 0 to 1 over 900 updates
    assert set(anim.beta[0]) == {4.0}
    assert anim.beta[1][0] == 0.0 and anim.beta[1][450] == 0.5 and anim.beta[1][-1] == 1.0


# ---------------------------------------------------------------------- 65 vq-vae-codebook
@needs("vq-vae-codebook")
def test_vq_study_remedies_keep_the_codebook_in_use_and_the_control_collapses():
    _anim, study = built("vq-vae-codebook")
    d = sg.arrays("vq-vae-codebook")
    used = (d["counts"] > 0).sum(-1)
    assert used.shape == (2, len(sg.VQ_SEEDS)) and len(sg.VQ_SEEDS) >= 3 and disjoint(d)
    assert np.array_equal(study.panels[3].data["values"], used)
    # with the remedies most of the codebook is used; without them a handful: every
    # seed against every seed, by a wide margin, and the reconstructions pay for it
    assert used[0].min() > used[1].max() + sg.VQ_CODES // 2
    assert used[0].min() >= 0.9 * sg.VQ_CODES
    assert d["mse"][0].max() < d["mse"][1].min() - 0.01
    assert (d["perplexity"][0] > d["perplexity"][1] + 2).all()
    assert (d["in_use"][0, :, -1] > d["in_use"][1, :, -1] + sg.VQ_CODES // 2).all()
    assert (d["counts"].sum(-1) == 1000 * sg.VQ_GRID ** 2).all()
    assert any("no collapse" in tried for tried in study.metadata["designs_tried"])
    # the saved reconstructions, code numbers and tiles are the saved model's
    held = sg.vq_data()[1]
    assert np.array_equal(sg.vq_unpatch(held[:12]), d["original"])
    assert np.array_equal(sg.vq_unpatch(sg.vq_patches(d["original"].reshape(12, -1))),
                          d["original"])
    for arm in ("remedies_", "control_"):
        enc, dec = sg.vq_split(M.weights(d, arm + "weights_"))
        _h, ids, out = sg.vq_apply(enc, dec, d[arm + "codes"], held[:12].reshape(-1, 16))
        assert np.array_equal(ids.reshape(12, 7, 7), d[arm + "ids"])
        assert np.allclose(sg.vq_unpatch(out.reshape(12, 49, 16)), d[arm + "recon"])
        assert np.allclose(sg.sigmoid(toy.forward(dec, d[arm + "codes"])[0]), d[arm + "tiles"])


@needs("vq-vae-codebook")
def test_vq_animation_draws_the_studys_codes_and_the_slope_is_the_gradient():
    anim, study = built("vq-vae-codebook")
    d = sg.arrays("vq-vae-codebook")
    used = (d["counts"] > 0).sum(-1)
    q = anim.q
    patch = anim.patches[0, anim.one]
    assert q["k"] == d["remedies_ids"][0].ravel()[anim.one]
    assert np.array_equal(q["z"], d["remedies_codes"][q["k"]])
    assert q["k"] == int(((q["h"][None] - d["remedies_codes"]) ** 2).sum(1).argmin())
    # the slope copied across the snap is the gradient of that patch's own loss at the code
    def loss(code):
        logits = toy.forward(anim.dec, code[None])[0][0]
        return float(np.mean(np.logaddexp(0, logits) - patch * logits))
    for axis in range(3):
        step = np.zeros(sg.VQ_LATENT)
        step[axis] = 1e-6
        slope = (loss(q["z"] + step) - loss(q["z"] - step)) / 2e-6
        assert slope == pytest.approx(q["dz"][axis], rel=1e-4, abs=1e-9)
    assert has(shown(anim, 2, 0.9), "(%.4f, %.4f, ...)" % tuple(q["dz"][:2]))
    # every patch the first phase walks through has ink, and is shown with its own code
    assert all(anim.patches[0, i].max() > 0.3 for i in anim.walk)
    first = shown(anim, 0, 0.06 + 0.88 * 0.9 / len(anim.walk))
    assert has(first, "nearest: code %d" % d["remedies_ids"][0].ravel()[anim.walk[0]])
    # the code sheet is the decoder's output for each code; unused codes can be greyed
    sheet = M.code_sheet(d["control_tiles"], dim=anim.dead[1])
    assert sheet.shape == (41, 41, 3) and len(anim.dead[1]) == sg.VQ_CODES - used[1, 0]
    assert np.allclose(M.code_sheet(d["remedies_tiles"])[1:5, 1:5],
                       M.shade(d["remedies_tiles"][0].reshape(4, 4)))
    assert has(shown(anim, 3, 0.95), "%d of %d codes ever chosen" % (used[1, 0], sg.VQ_CODES))
    last = shown(anim, 4, 0.95)
    assert has(last, "without: %s of %d used" % (", ".join(str(v) for v in used[1]),
                                                  sg.VQ_CODES))
    assert has(last, "error per pixel: %.3f with, %.3f to %.3f without" % (
        d["mse"][0].mean(), d["mse"][1].min(), d["mse"][1].max()))
    assert np.array_equal(study.panels[0].data["values"][4:8], d["remedies_recon"][:4])


# ---------------------------------------------------------------------- 66 autoregressive-pixels
def test_pixel_model_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    dilations = (1, 2, 1)
    p = {k: v.astype(float) + 0.05 * rng.normal(size=v.shape)
         for k, v in sg.pix_init(rng, dilations, channels=3).items()}
    x = (rng.random((2, 6, 6, 1)) < 0.4).astype(float)
    _loss, g = sg.pix_loss_grads(p, x, dilations)
    for key in p:
        index = tuple(rng.integers(0, n) for n in p[key].shape)
        old = p[key][index]
        p[key][index] = old + 1e-6
        up = sg.pix_loss_grads(p, x, dilations)[0]
        p[key][index] = old - 1e-6
        down = sg.pix_loss_grads(p, x, dilations)[0]
        p[key][index] = old
        assert (up - down) / 2e-6 == pytest.approx(g[key][index], rel=1e-4, abs=1e-8), key


def test_pixel_masks_reach_and_blind_spot_are_exact():
    side = 9
    order = np.arange(side * side).reshape(side, side)
    two = sg.pix_seen(sg.PIX_DEEP, side)
    plain = sg.plain_seen(sg.PIX_DEEP, side)
    deep = sg.plain_seen((1,) * (4 * side), side)
    for r, c in ((0, 0), (4, 2), (8, 8), (6, 0)):
        earlier = order < r * side + c
        # nothing reads itself or the future; the two-stream model reads all a stack does
        assert not (two[r, c] & ~earlier).any() and not (plain[r, c] & ~earlier).any()
        assert not (plain[r, c] & ~two[r, c]).any()
        # a single stack's blind spot at any depth: above and more to the right than up
        blind = {(i, j) for i in range(r) for j in range(side) if j - c > r - i}
        assert {tuple(v) for v in np.argwhere(earlier & ~deep[r, c])} == blind
    # one first-layer mask reads the four neighbours of MASK_A
    assert {tuple(v) for v in np.argwhere(sg.plain_seen((1,), side)[4, 4])} == {
        (3, 3), (3, 4), (3, 5), (4, 3)}
    assert (two[6, 0] & ~deep[6, 0]).sum() > 0          # the second stream closes some of it


@needs("autoregressive-pixels")
def test_pixel_study_six_layers_beat_one_layer_and_independent_pixels():
    _anim, study = built("autoregressive-pixels")
    d = sg.arrays("autoregressive-pixels")
    held, independent = d["held_bits"], float(d["independent"])
    assert held.shape == (2, len(sg.PIX_SEEDS)) and len(sg.PIX_SEEDS) >= 3 and disjoint(d)
    # held-out bits per pixel: six layers < one layer < each pixel on its own, every seed
    assert held[0].max() < held[1].min() - 0.05
    assert held[1].max() < independent - 0.03
    assert np.allclose(study.panels[3].data["values"],
                       [independent, held[1].mean(), held[0].mean()])
    x, test = sg.pix_data()[:2]
    assert set(np.unique(x)) == {0.0, 1.0} and len(x) == 50000 and len(test) == 2000
    p = M.weights(d)
    assert held[0, 0] == pytest.approx(sg.pix_bits(p, test, sg.PIX_DEEP), rel=1e-5)
    q = np.clip(x.mean(0), 1e-3, 1 - 1e-3)
    assert independent == pytest.approx(
        float(np.mean(-(test * np.log2(q) + (1 - test) * np.log2(1 - q)))))
    # samples carry about a real digit's ink, where the first study's were nearly blank
    ink = d["samples"].reshape(16, -1).sum(1)
    assert ink.min() >= 8 and 0.6 < ink.mean() / test.reshape(len(test), -1).sum(1).mean() < 1.5


@needs("autoregressive-pixels")
def test_pixel_chances_are_causal_and_the_samples_are_their_draws():
    anim, study = built("autoregressive-pixels")
    d = sg.arrays("autoregressive-pixels")
    side = sg.PIX_SIDE
    assert np.array_equal(anim.samples[:8], study.panels[2].data["values"])
    assert np.array_equal(d["samples"], (d["draws"] < d["chances"]).astype(float))
    # the chance a pixel was drawn with depends on earlier pixels only
    for k in (0, 17, 100, 195):
        r, c = divmod(k, side)
        cut = d["samples"][:4].copy().reshape(4, -1)
        cut[:, k:] = 0.0
        again = M.pixel_chances(anim.p, cut.reshape(4, side, side))
        assert np.allclose(again[:, r, c], d["chances"][:4, r, c], atol=1e-5)
    # measured on the trained model: a blind-spot pixel the second stream reads does
    # change the prediction, and a later pixel does not
    r, c = anim.TARGET
    base = M.pixel_chances(anim.p, anim.real[:1])[0, r, c]
    closed = np.argwhere(anim.never & anim.two)
    assert len(closed) == anim.closed > 0
    moved = []
    for i, j in closed:
        flipped = anim.real[:1].copy()
        flipped[0, i, j] = 1 - flipped[0, i, j]
        moved.append(abs(M.pixel_chances(anim.p, flipped)[0, r, c] - base))
    assert max(moved) > 1e-4
    later = anim.real[:1].copy()
    later[0, r, c:] = 1 - later[0, r, c:]
    later[0, r + 1:] = 1 - later[0, r + 1:]
    assert M.pixel_chances(anim.p, later)[0, r, c] == pytest.approx(base, abs=1e-7)
    assert np.array_equal(study.panels[0].data["values"],
                          anim.two.astype(int) + (anim.two & anim.views[-1]))
    masks = shown(anim, 1, 0.65)
    assert has(masks, "%d: the blind spot" % anim.never.sum())
    assert has(shown(anim, 1, 0.95), "with %d of the %d" % (anim.closed, anim.never.sum()))
    held = d["held_bits"]
    assert has(shown(anim, 4, 0.9), "%.3f bits per pixel with six layers, %.3f with one, %.3f"
               % (held[0].mean(), held[1].mean(), float(d["independent"])))

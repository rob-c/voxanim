"""The numbers behind animations 265-268 (voxanim.study_blocks_zc): an autoencoder's blocks
wired again as a GAN. Everything comes from voxanim.study_aegan (the numpy twin of
examples/ae_gan.py); the claims shown on screen are checked against it here."""

from __future__ import annotations

import math

import numpy as np
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import study_aegan as A
from voxanim import study_blocks_zc as Z


# ---------------------------------------------------------------------- no data needed
def test_the_counts_are_in_times_out_plus_out():
    c = Z.counts()
    assert (c["enc1"], c["enc2"], c["dec1"], c["dec2"], c["head"]) == (
        200_960, 4_112, 4_352, 201_488, 257)
    assert (c["encoder"], c["decoder"], c["judge"]) == (205_072, 205_840, 201_217)
    assert c["generator"] == c["decoder"]                 # the same block
    assert c["judge"] - c["head"] == c["enc1"]            # the encoder's body, kept
    # the layers as the script writes them
    assert "nn.Linear(784, 256), nn.ReLU()," in B.code_line("ag-blocks", "784, 256")
    assert B.code_line("ag-rewire", "nn.Linear") == "    nn.Linear(256, 1))"
    assert B.code_lines("ag-rewire")[0] == "generator = decoder"


def test_one_pixels_score_is_lowest_at_the_share_of_ink():
    s = Z.pixel_scores()
    assert [round(v, 3) for v in s] == [1.204, 0.693, 1.204]
    assert math.isclose(s[1], math.log(2)) and math.isclose(s[0], s[2])
    grid = np.linspace(0.01, 0.99, 99)
    best = grid[np.argmin(Z.pixel_scores(0.5, grid))]
    assert math.isclose(best, 0.5)
    best = grid[np.argmin(Z.pixel_scores(0.3, grid))]      # ink in 30%: paint 0.3
    assert math.isclose(best, 0.3)


def test_the_tiles_drawn_are_corners_of_the_weights():
    W = np.arange(256 * 784, dtype=float).reshape(256, 784) - 1000.0
    p = Z.patch(W, 12, centre=True)
    assert p.shape == (12, 5)
    assert p[0, 0] == abs(W[0, 14 * 28 + 12]) and p[-1, -1] == abs(W[255, 14 * 28 + 16])
    q = Z.patch(np.ones((16, 256)), 5)
    assert q.shape == (5, 5) and (q == 1).all()


def test_helpers():
    assert Z.pct(0.0941) == "9.4%" and Z.pct(0.928) == "92.8%"
    np.testing.assert_allclose(Z.chunked(np.arange(300.0), 100), [49.5, 149.5, 249.5])
    assert Z.grey_mask(np.array([0.0, 0.2, 0.5, 0.8, 1.0])).tolist() == [
        False, False, True, False, False]


# ---------------------------------------------------------------------- 265
@needs_mnist
def test_rewire():
    s = Z.rewire().validate()
    m = s.metadata
    enc, dec, j0 = A.blocks("enc"), A.blocks("dec"), A.blocks("judge0")
    # the judge's body is the encoder's, number for number; its head is new
    assert np.array_equal(j0["0.weight"], enc["0.weight"])
    assert np.array_equal(j0["0.bias"], enc["0.bias"])
    np.testing.assert_array_equal(j0["2.weight"], A.judge_head()["2.weight"])
    assert m["kept"] == sum(enc[k].size for k in ("0.weight", "0.bias")) == 200_960
    assert m["left_out"] == sum(enc[k].size for k in ("2.weight", "2.bias")) == 4_112
    assert m["counts"]["encoder"] == A.summary()["encoder_numbers"]
    assert m["counts"]["decoder"] == A.summary()["decoder_numbers"]
    assert m["counts"]["head"] == A.summary()["judge_new"]
    # the tiles drawn are the real weights' corners
    np.testing.assert_array_equal(m["patches"]["enc1"],
                                  Z.patch(enc["0.weight"], 12, centre=True))
    np.testing.assert_array_equal(m["patches"]["dec2"], Z.patch(dec["2.weight"].T, 16))
    np.testing.assert_array_equal(m["patches"]["head"], Z.patch(j0["2.weight"], 1))
    # the digit, its code, its rebuild; the decoder's own picture of a made-up code
    x = A.pixels()[1][0]
    np.testing.assert_allclose(m["code"], A.encode(enc, x[None])[0], atol=1e-6)
    np.testing.assert_allclose(m["rebuilt"].ravel(), A.decode(dec, A.encode(enc, x[None]))[0],
                               atol=1e-6)
    assert (m["digit_label"], m["rebuilt_read"], m["real_label"]) == (7, 7, 2)
    assert m["digit_loss_text"] == "0.07" and len(m["code"]) == 16
    np.testing.assert_allclose(m["made"].ravel(), A.decode(dec, A.run()["fixed"][:1])[0],
                               atol=1e-6)
    np.testing.assert_array_equal(m["made_code"], A.run()["fixed"][0])
    assert "205,840" in s.subtitle and "200,960" in s.subtitle and "257" in s.subtitle
    assert any("left out" in x for x in m["limitations"])


# ---------------------------------------------------------------------- 266
@needs_mnist
def test_blurry():
    s = Z.blurry().validate()
    m = s.metadata
    assert [m["grey_text"][k] for k in m["kinds"]] == ["6.0%", "9.4%", "9.1%", "7.4%"]
    assert [m["sure_text"][k] for k in m["kinds"]] == ["0.94", "0.94", "0.74", "0.84"]
    g = m["grey"]
    assert g["real"] < g["after"] < g["before"] < g["rebuilt"]
    # the pictures: the first three held-out digits, their rebuilds, and the decoder on the
    # same three made-up codes before (the decoder) and after (the generator) the game
    r = A.run()
    z = A.noise(r["mean"], r["spread"])[:3]
    np.testing.assert_allclose(m["shown"]["before"].reshape(3, -1),
                               A.decode(A.blocks("dec"), z), atol=1e-6)
    np.testing.assert_allclose(m["shown"]["after"].reshape(3, -1),
                               A.decode(A.blocks("gen"), z), atol=1e-6)
    np.testing.assert_array_equal(m["shown"]["real"].reshape(3, -1), A.pixels()[1][:3])
    assert m["scores_text"] == ["1.20", "0.69", "1.20"]
    assert "6.0%" in s.subtitle and "7.4%" in s.subtitle


# ---------------------------------------------------------------------- 267
@needs_mnist
def test_rebuild():
    s = Z.rebuild().validate()
    m = s.metadata
    assert (m["rebuild_right_n"], m["real_right_n"]) == (928, 949)
    assert (m["rebuild_right_text"], m["real_right_text"]) == ("92.8%", "94.9%")
    after = m["made_counts"]["after"]
    assert sum(after) == 1000 and min(after) > 0           # every digit made
    assert (m["after_min"], m["after_min_digit"], m["after_max"], m["after_max_digit"]) == (
        63, 5, 135, 8)
    assert m["pair_labels"] == m["pair_read"] == [7, 2, 1, 0]
    # the 16 shown are the fixed codes' pictures after the game, as they are
    np.testing.assert_array_equal(m["made"].reshape(16, -1), A.run()["made6000"])
    assert m["made_read"] == A.read(A.run()["made6000"])[0].tolist()
    np.testing.assert_array_equal(m["made_code"], A.run()["fixed"][0])
    assert "92.8%" in s.subtitle and "63 to 135" in s.subtitle


# ---------------------------------------------------------------------- 268
@needs_mnist
def test_losses():
    s = Z.losses().validate()
    m = s.metadata
    assert (m["ae_first_text"], m["ae_last_text"]) == ("0.69", "0.10")
    assert m["lj_text"] == ("1.43", "1.20") and m["lg_text"] == ("0.67", "0.88")
    assert len(m["ae_loss"]) == A.AE_STEPS and len(m["lj_chunks"]) == A.GAN_STEPS // 100
    assert math.isclose(m["lj_chunks"][0], m["lj_first"])
    assert math.isclose(m["lg_chunks"][-1], m["lg_last"])
    assert math.isclose(m["lj_chunks"].mean(), float(A.run()["lj"].mean()), rel_tol=1e-9)
    # a judge that says 0.5 to everything: exact
    assert round(m["coin_judge"], 2) == 1.39 and round(m["coin_gen"], 2) == 0.69
    assert m["lj_last"] < m["lj_first"] and m["lg_last"] > m["lg_first"]   # they pull
    assert m["snaps"] == [0, 500, 2000, 6000]
    np.testing.assert_array_equal(m["snap_pics"][0].reshape(2, -1), A.run()["made0"][:2])
    assert "0.69 to 0.10" in s.subtitle

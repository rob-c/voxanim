"""The numbers behind animations 275-276 (voxanim.study_blocks_zh): event weights and scale
factors on made-up events. Everything comes from voxanim.study_weights (the numpy twin of
examples/weights_and_scale_factors.py); the strings shown on screen are checked against it here.
No dataset is needed: the data are made up by the script."""

from __future__ import annotations

import math

import numpy as np

from voxanim import blocks as B
from voxanim import study_blocks_zh as Z
from voxanim import study_weights as W
from voxanim.next_catalog import build_study


def weights():
    return build_study("study_blocks_zh", "event-weights").metadata


def scale():
    return build_study("study_blocks_zh", "scale-factors").metadata


def test_the_numbers_are_written_as_the_screen_writes_them():
    assert Z.pct(0.01795) == "1.8%" and Z.pct(5000 / 37000) == "13.5%"
    assert Z.three(0.80530) == "0.805" and Z.three(0.92487) == "0.925"
    assert [Z.weight_text(v) for v in (0.25, 0.4, 1.2, 2.0)] == ["0.25", "0.4", "1.2", "2"]


def test_both_studies_have_four_finite_panels_and_say_what_is_made_up():
    for name in ("event-weights", "scale-factors"):
        s = build_study("study_blocks_zh", name).validate()
        assert len(s.panels) == 4
        assert "made-up" in s.metadata["source"]
        assert any("made-up" in t or "made up" in t for t in s.metadata["limitations"])


# ---------------------------------------------------------------------- 275
def test_the_table_is_cross_section_times_luminosity_over_events():
    m = weights()
    assert m["names"] == ["signal", "bkg 1", "bkg 2"]
    assert m["sigma_text"] == ["50", "200", "120"] and m["n_text"] == ["20,000", "50,000",
                                                                       "10,000"]
    assert m["weights_text"] == ["0.25", "0.4", "1.2"]
    for s, n, w in zip(m["sigma"], m["n"], m["weights"], strict=True):
        assert w == s * m["L"] / n
    assert m["sums_text"] == m["expected_text"] == ["5,000", "20,000", "12,000"]
    assert m["worked"] == "50 × 100 ÷ 20,000 = 0.25"
    assert m["printed_hist"] == "signal 5000\nbkg 1 20000\nbkg 2 12000"
    assert m["n_total"] == 80000 and m["expected_total"] == 37000
    np.testing.assert_allclose(np.sum(m["weighted"], 1), [5000, 20000, 12000])
    assert m["counts_max"] < 18000 and m["weighted_max"] < 9000    # the chart's two ranges


def test_the_loss_and_the_score():
    m = weights()
    assert (m["mean_unweighted_text"], m["mean_weighted_text"]) == ("0.250", "0.135")
    assert (m["share_unweighted_text"], m["share_weighted_text"]) == ("25.0%", "13.5%")
    assert math.isclose(m["trained"]["first_loss"], math.log(2))
    assert m["trained"]["last_loss"] < 0.2
    s = np.asarray(m["score_weighted"])
    assert (np.diff(s) > 0).all() and s[0] < 0.01 and s[-1] > 0.99
    assert m["half_unweighted_text"] == "0.875" and m["half_weighted_text"] == "0.933"
    assert m["code_loss"] == list(B.code_lines("ws-loss"))
    assert "loss = (w * ce).sum() / w.sum()" in m["code_loss"]


def test_the_efficiency_rows():
    m = weights()
    assert m["eff_counted_row"] == "counted: 1,077 ÷ 60,000 = 1.8%"
    assert m["eff_weighted_row"] == "weighted: 1,124 ÷ 32,000 = 3.5%"
    assert m["printed_eff"] == "0.01795\n0.0351125"
    assert m["cut_text"] == "0.915"


def test_the_negative_weights_rows():
    m = weights()
    assert m["neg_rows"] == ["8,000 × (+2), 2,000 × (−2)", "Σw = 12,000, as before",
                             "N_eff = (Σw)² ÷ Σw² = 3,600"]
    assert m["n_eff_text"] == "3,600"
    assert math.isclose(m["err_ratio"], 5 / 3)
    assert min(m["neg"]["hist_neg"]) > -2000 and max(m["neg"]["hist_pos"]) < 6500  # in range
    assert min(m["neg"]["net"]) >= 0                     # no bin goes negative here


# ---------------------------------------------------------------------- 276
def test_the_control_region_and_the_scale_factor():
    m = scale()
    assert m["signal_kept"] == 16000 and m["signal_n"] == 20000
    assert m["eff_mc_row"] == "16,106 ÷ 20,000 = 0.805"
    assert m["eff_data_row"] == "3,724 ÷ 5,000 = 0.745"
    assert m["sf_row"] == "0.745 ÷ 0.805 = 0.925" and m["sf_err_text"] == "0.008"
    # the data really are wider: less in the middle, more in the tails
    mc, data = np.asarray(m["probe_mc_shares"]), np.asarray(m["probe_data_shares"])
    assert data.max() < mc.max() and data[:6].sum() > mc[:6].sum()
    assert max(mc.max(), (data + np.asarray(m["probe_data_err"])).max()) < 0.21  # the chart
    assert max(m["signal_counts"]) < 4200                 # the first chart's range


def test_the_signal_region():
    m = scale()
    assert (m["predicted_text"], m["corrected_text"], m["observed_text"]) == (
        "5,124", "4,823", "4,739")
    assert (m["data_err_text"], m["corrected_err_text"]) == ("69", "57")
    assert (m["signal_text"], m["signal_scaled_text"], m["background_text"]) == (
        "4,000", "3,699", "1,124")
    assert m["excess_text"] == "385" and m["pull_before_text"] == "4.6"
    assert m["pull_after_text"] == "0.9"
    sr = m["sr"]
    for v in (sr["predicted"], sr["corrected"], sr["observed"]):    # on the number line
        assert 4500 < v < 5250
    assert sr["predicted"] - sr["predicted_err"] > sr["observed"] + sr["data_err"]
    assert abs(sr["corrected"] - sr["observed"]) < math.hypot(sr["corrected_err"],
                                                              sr["data_err"])


def test_the_corrected_count_is_not_the_shape():
    """The limitation said on screen: a scale factor fixes the count passing, not the shape;
    above x = 1.2 the wider data have more events than the corrected simulation."""
    c = W.control()
    d = W.made_up_data()["data"]
    x, w = W.simulated(), W.weights()
    above = int((d > 1.2).sum())
    pred = sum(((c["sf"] if k == "signal" else 1.0) * w[k][x[k] > 1.2]).sum() for k in W.NAMES)
    assert above > 2 * pred

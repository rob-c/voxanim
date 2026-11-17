"""Animations 110-114 (voxanim.mech_random) and their experiments (voxanim.study_random):
the layout is clean, the loop closes, the numbers on screen are the study's, the small
generators are the published ones, each study's claim holds as a relative claim, and the
recorded arrays are what they say they are.

The first four animations need no data and are built here from an empty cache. The
measurements of this machine (timer jitter, the operating system's bytes) differ on every
machine: their values are never asserted, only shapes, ranges and what must hold anywhere.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from conftest import data_marks, needs_text

from voxanim import cache, layout, presets, qa
from voxanim import mech_random as M
from voxanim import study_random as S
from voxanim.next_catalog import NO_DATA, TEXT, build_study

NO_DATA_NAMES = ("pseudo-random-generators", "seeds-and-streams", "sampling-a-distribution",
                 "entropy-sources")
NAMES = NO_DATA_NAMES + ("measuring-entropy",)
EACH = [pytest.param(n, marks=data_marks(n)) for n in NAMES]


def study(name):
    return build_study("study_random", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, t):
    return [x.s for x in anim.scene(t).texts]


def mid(anim, k, at=0.97):
    a, b = anim.spans[k]
    return min(a + (b - a) * at, b - anim.FADE - 0.05)


# ---------------------------------------------------------------------- all five
def test_registered_with_what_they_need():
    assert set(NO_DATA_NAMES) <= NO_DATA and "measuring-entropy" in TEXT
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    anim = presets.get(name).build()
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for a, b in anim.spans:                           # every phase draws things, not only text
        assert len(anim.scene((a + b) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_evidence_is_labelled(name):
    anim = presets.get(name).build()
    for sub in subtitles(anim):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    kinds = (M.EXACT, M.STATS, M.LOCAL, M.WRITTEN, M.MIXED, M.STATS + "; right: local")
    for k in range(len(anim.PHASES)):                 # every phase says what kind of evidence
        assert any(t in kinds for t in texts(anim, mid(anim, k, 0.5)))


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    assert len(s.panels) == 4 and s.metadata["designs_tried"] and s.metadata["limitations"]
    assert s.metadata["evidence"]


def test_no_data_studies_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NO_DATA_NAMES:                        # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = sorted(p.name.split("-")[1] for p in tmp_path.iterdir())
    assert made == ["generators", "local", "sampling", "seeds", "sources"]


# ---------------------------------------------------------------------- the generators
def test_generators_are_the_published_ones():
    assert int(S.pcg32_reference(42, 54, 1)[0]) == 0xA15C02B7       # pcg32-demo's first output
    assert S.xs_step(2463534242) == 723471715                       # Marsaglia (2003), xor32
    assert S.splitmix64(1234567) == 6457827717110365317
    assert [int(v) for v in S.lcg_states(1, 3, *S.RANDU)] == [65539, 393225, 1769499]
    assert S.lcg_step(0, *S.LCG32) == 1013904223
    for x in (0, 1, 2, 3, 1 << 63):
        stages = S.splitmix64_stages(x)
        assert stages[0] == x and stages[-1] == S.splitmix64(x)
        assert int(S.splitmix64_array(np.array([x], np.uint64))[0]) == S.splitmix64(x)


def test_doubling_and_jumping_equal_stepping():
    for spec in (S.LCG32, S.RANDU, (S.PCG_MULT, S.PCG_INC, 64), (S.PCG16_MULT, S.PCG16_INC, 32)):
        x, seq = 12345, []
        for _ in range(300):
            x = S.lcg_step(x, *spec)
            seq.append(x)
        assert [int(v) for v in S.lcg_states(12345, 300, *spec)] == seq
        a, c = S.lcg_power(*spec, 300)
        assert (a * 12345 + c) % (1 << spec[2]) == seq[-1]
    x, seq = 1, []
    for _ in range(300):
        x = S.xs_step(x)
        seq.append(x)
    assert [int(v) for v in S.xs_states(1, 300)] == seq and S.xs_jump(1, 300) == seq[-1]
    # PCG32's output function, bit by bit on Python integers
    old = int(S.lcg_states(7, 5, S.PCG_MULT, S.PCG_INC, 64)[-1])
    word = (((old >> 18) ^ old) >> 27) & S.M32
    rot = old >> 59
    assert int(S.pcg32_parts(np.array([old], np.uint64))[2][0]) == (
        (word >> rot) | (word << ((32 - rot) & 31))) & S.M32
    state = np.arange(1 << 12, dtype=np.uint32) * np.uint32(2654435761)
    x = (((state >> 10) ^ state) >> 12) & 0xFFFF
    r = state >> 28
    assert np.array_equal(S.pcg16_output(state), ((x >> r) | (x << ((16 - r) & 15))) & 0xFFFF)


def test_estimators_on_known_inputs():
    counter = (np.arange(1 << 12) & 255).astype(np.uint8)
    assert S.byte_entropy(counter) == 8.0 and S.min_entropy(counter) == 8.0
    assert S.conditional_entropy(counter, 1) < 1e-6
    assert S.shannon([1, 1, 2]) == 1.5 and S.shannon([5]) == 0.0
    assert np.isclose(S.coin_entropy(0.5), 1.0) and S.coin_entropy(1.0) == 0.0
    assert np.isclose(S.chi_square_z(counter), -255 / np.sqrt(510))
    assert S.runs_z(counter) < -50 and S.serial_z(counter) > 50
    assert list(S.von_neumann([0, 1, 1, 0, 1, 1, 0, 0, 1])) == [0, 1]
    assert list(S.inverse_cdf([0.0, 0.3999, 0.4, 0.9499, 0.9501, 0.9999], S.TOKEN_P)) == [
        0, 0, 1, 3, 4, 4]
    assert S.total_variation([4, 0], [0.5, 0.5]) == 0.5
    assert S.sequence_period(np.tile([1, 2, 3, 1, 2, 3], 4), [1, 2, 3, 6]) == 3
    assert S.short_period(np.tile([1, 0, 0, 1, 1], 60)) == 5
    assert S.short_period(np.random.default_rng(0).integers(0, 2, 4096), 1024) == 0


# ---------------------------------------------------------------------- 110
def test_generators_claims():
    s, d = study("pseudo-random-generators"), S.prg_data()
    assert all(v["proved"] for v in s.metadata["periods"].values())
    assert [v["period"] for v in s.metadata["periods"].values()] == [
        "16", "2^29", "2^32", "2^32 - 1", "2^64"]
    cycle = [int(v) for v in s.panels[0].data["ys"][0]]
    assert sorted(cycle[:-1]) == list(range(16)) and cycle[0] == cycle[-1] == 1
    assert [int(v) for v in s.panels[0].data["short_cycle"]] == [1, 9, 1]
    assert [int(v) for v in s.panels[1].data["low_bit_period"]] == [2 ** (k + 1) for k in range(16)]
    stats = np.asarray(s.panels[2].data["all_seeds"])           # (generator, seed, statistic)
    randu, lcg, good = stats[0], stats[1], stats[2:]
    ideal = float(s.panels[2].data["ideal_cells"])
    assert np.abs(stats[..., 0]).max() < 0.01                   # serial correlation sees nothing
    assert randu[:, 1].max() < 0.3 and np.abs(stats[1:, :, 1] - ideal).max() < 0.005
    assert (randu[:, 2] == 1).all() and (randu[:, 3] == 0).all()        # lowest bit constant
    assert (lcg[:, 2] == 2).all() and (lcg[:, 3] == 1).all()            # ... alternating
    assert (good[..., 2] == 0).all() and np.abs(good[..., 3] - 0.5).max() < 0.005
    assert len(d["randu_planes"]) == 15


def test_generators_data_is_what_it_says():
    d = S.prg_data()
    x, rows = 1, []
    for _ in range(len(d["lcg_rows"])):
        x = (1664525 * x + 1013904223) % 2 ** 32
        rows.append(x)
    assert [int(v) for v in d["lcg_rows"]] == rows and rows[0] == 1015568748
    x = [int(v) for v in d["xs_demo"]]
    assert x[-1] == S.xs_step(x[0]) and x[2] == x[0] ^ x[1] and x[4] == x[2] ^ x[3]
    old, mixed, rot, word = (int(v) for v in d["pcg_demo"])
    assert rot == old >> 59 and word == ((mixed >> rot) | (mixed << (32 - rot))) & S.M32
    x = S.lcg_states(1, 6002, *S.RANDU).astype(np.int64)        # every triple on a plane
    assert np.allclose(d["randu_points"], x / 2.0 ** 31)
    k = 9 * x[:-2] - 6 * x[1:-1] + x[2:]
    assert (k % 2 ** 31 == 0).all() and set(k // 2 ** 31) <= {int(v) for v in d["randu_planes"]}
    u = S.stream("randu", 1, 5000)[0]
    assert np.isclose(S.cells_hit(u, 4), 1.0) and S.cells_hit(np.zeros(10), 4) == 1 / 64


def test_generators_numbers_on_screen():
    s = study("pseudo-random-generators")
    anim = M.PseudoRandomGenerators(s)
    sub = subtitles(anim)
    stats = np.asarray(s.panels[2].data["values"])
    assert "one of 15 planes, and reach %.0f%% of a cube's 262,144 cells" % (
        100 * stats[0, 1]) in sub[3]
    assert "period 4,294,967,296" in sub[2] and "comes round every 256" in sub[2]
    assert "back after 2 steps" in sub[1]
    shown = texts(anim, mid(anim, 5))
    for g in range(5):
        assert "%+.4f" % stats[g, 0] in shown and "%.0f%%" % (100 * stats[g, 1]) in shown
    assert {"2²⁹", "2³²", "2³² − 1", "2⁶⁴", "repeats every 2", "never changes"} <= set(shown)
    assert "15 planes" in texts(anim, mid(anim, 3))
    assert "5 × 10 + 3 = 53" in texts(anim, mid(anim, 0))


# ---------------------------------------------------------------------- 111
def test_seeds_claims():
    s = study("seeds-and-streams")
    equal = np.asarray(s.panels[0].data["values"])
    assert equal[0] == equal[3] == 1.0 and equal[1] == equal[4] == 0.0
    assert abs(equal[2] - 0.5) < 0.005
    corr = np.asarray(s.panels[1].data["correlation"])
    gap = np.asarray(s.panels[1].data["adjacent_gap"])
    assert corr[0] > 0.999 and np.abs(corr[1:]).max() < 0.25
    assert gap[0] < 0.001 and gap[1:].min() > 0.25
    assert list(s.panels[2].data["relation"]) == [1.0, 0.0, 0.0]
    assert np.abs(s.panels[2].data["pearson"]).max() < 0.02     # which Pearson does not see
    weak, hashed = np.asarray(s.panels[2].data["values"]), np.asarray(s.panels[2].data["hashed"])
    assert list(weak[:, 0]) == [0.0, 1.0, 0.0]                  # the lowest bit is locked
    assert np.abs(hashed - 0.5).max() < 0.01
    assert np.abs(s.panels[2].data["numpy"] - 0.5).max() < 0.01
    m = s.metadata
    assert abs(m["splitmix64_bits_changed_per_seed_bit"]["mean"] - 32) < 0.5
    same, jumped, spawned, one = s.panels[3].data["values"]
    assert same == s.panels[3].data["one_worker"][0]            # four workers, no gain
    assert jumped < 0.6 * one and spawned < 0.6 * one and same > 0.9 * one
    wc = np.asarray(s.panels[3].data["worker_correlation"])
    assert wc[0] > 0.999 and np.abs(wc[1:]).max() < 0.01


def test_seeds_data_is_what_it_says():
    d = S.seeds_data()
    assert np.array_equal(d["same"][0], d["same"][1])
    assert np.array_equal(d["same"][2], S.pcg32(S.splitmix64(43), 8))
    lcg = d["streams"][0]
    assert int(lcg[1][0]) == S.lcg_step(2, *S.LCG32)
    assert int(lcg[0][1]) == S.lcg_step(S.lcg_step(1, *S.LCG32), *S.LCG32)
    assert np.allclose(d["first"][0][:3], [S.lcg_step(k, *S.LCG32) / 2 ** 32 for k in (1, 2, 3)])
    assert [int(v) for v in d["hashed"]] == [S.splitmix64(k) for k in (1, 2, 3)]
    # the jumped workers: worker w starts w * 2^40 steps along worker 0's stream
    state = S.splitmix64(1000)
    a, c = S.lcg_power(S.PCG_MULT, S.PCG_INC, 64, 2 * S.PAR_JUMP)
    darts = S.unit(S.pcg32((a * state + c) & S.M64, 2 * S.PAR_DARTS)).reshape(-1, 2)
    assert np.allclose(d["darts"][2], darts) and np.isclose(
        S.pi_estimate(darts.ravel()), d["pi"][1, 0, 2])
    assert S.lcg_power(S.PCG_MULT, S.PCG_INC, 64, 0) == (1, 0)
    assert np.all(d["pi"][0] == d["pi"][0][:, :1])              # one seed: the same answer


def test_seeds_numbers_on_screen():
    s = study("seeds-and-streams")
    anim = M.SeedsAndStreams(s)
    sub = subtitles(anim)
    assert "100,000 of 100,000 outputs equal" in sub[0] and "0 equal" in sub[0]
    assert "(correlation %.3f)" % s.panels[1].data["correlation"][0] in sub[1]
    assert "for 100% of 100,000 outputs" in sub[2]
    rms = s.panels[3].data["values"]
    assert "error %.3f, as one worker's" % rms[0] in sub[4]
    assert "%.3f and %.3f" % (rms[1], rms[2]) in sub[4]
    a, b = S.splitmix64(1), S.splitmix64(2)
    assert "until %d of 64 differ" % bin(a ^ b).count("1") in sub[3]
    shown = texts(anim, mid(anim, 4))
    assert {"%.3f" % v for v in rms} <= set(shown)
    lcg = S.seeds_data()["streams"][0]
    assert M.commas((2 * int(lcg[1][0]) - int(lcg[0][0])) % 2 ** 32) in texts(anim, mid(anim, 2))


# ---------------------------------------------------------------------- 112
def test_sampling_claims():
    s, d = study("sampling-a-distribution"), S.sampling_data()
    p = np.asarray(S.TOKEN_P)
    assert np.abs(s.panels[0].data["values"] - p).max() < 0.005
    assert abs(float(d["rej_rate"]) - 9 / 16) < 0.005
    mean, sd, within, corr = d["bm_stats"]
    assert abs(mean) < 0.01 and abs(sd - 1) < 0.01 and abs(within - 0.6827) < 0.005
    assert abs(corr) < 0.01
    err, pairs, _bad = np.asarray(s.panels[3].data["ys"])
    assert abs(s.metadata["error_slope"] + 0.5) < 0.03 and (np.diff(err) < 0).all()
    assert (np.diff(pairs) < 0).all()
    lcg = np.asarray(s.panels[3].data["pairs_lcg"])             # (seed, draws)
    ctl = np.asarray(s.panels[3].data["pairs_pcg_low_byte"])
    assert lcg[:, -1].min() > 5 * pairs[-1] and lcg[:, -1].min() > 3 * ctl[:, -1].max()
    assert np.abs(lcg[:, -1] - lcg[:, -5]).max() < 0.001        # stuck: more draws do not help
    assert s.metadata["lcg_low_byte_token_period"] == 256 and int(d["period_pcg"]) == 0
    assert (d["pair_lcg"] == 0).sum() >= 1 and (d["pair_good"] == 0).sum() == 0


def test_sampling_data_is_what_it_says():
    d = S.sampling_data()
    u = S.unit(S.pcg32_seeded(7, 24))
    assert np.allclose(d["token_u"], u) and np.array_equal(
        d["token_idx"], S.inverse_cdf(u, S.TOKEN_P))
    assert int(d["token_counts"].sum()) == S.SAMP_N
    assert np.array_equal(d["rej_keep"], d["rej_y"] < S.target(d["rej_x"]))
    assert np.isclose(S.target(1 / 3), S.BOUND)
    assert S.target(np.linspace(0, 1, 999)).max() <= S.BOUND
    z1, z2 = S.box_muller(d["bm_u"][:, 0], d["bm_u"][:, 1])
    assert np.allclose(d["bm_z"], np.stack([z1, z2], 1))
    anim = M.SamplingADistribution(study("sampling-a-distribution"))
    assert [int(c.sum()) for c in anim.prefix] == list(S.SAMP_GRID)
    x = S.lcg_states(1, 600, *S.LCG32)
    assert np.array_equal(d["tokens_lcg"], S.inverse_cdf(S.low_byte_unit(x), S.TOKEN_P)[:32])
    assert np.array_equal(S.low_byte_unit(x)[:256] * 256, np.asarray(x[:256] % 256, float))


def test_sampling_numbers_on_screen():
    s = study("sampling-a-distribution")
    anim = M.SamplingADistribution(s)
    sub = subtitles(anim)
    err, pairs, bad = np.asarray(s.panels[3].data["ys"])
    assert "%.3f, %.3f, %.3f, %.3f, %.3f" % tuple(s.panels[0].data["values"]) in sub[0]
    assert "%.1f%% were kept" % (100 * s.metadata["rejection"]["kept"]) in sub[1]
    assert "falls from %.3f at 10 draws to %.4f at 163,840" % (err[0], err[-1]) in sub[3]
    assert "stays at %.3f while numpy's falls to %.3f" % (bad[-1], pairs[-1]) in sub[4]
    assert "repeat every 256 draws" in sub[4]
    assert M.WRITTEN in texts(anim, mid(anim, 0))               # the probabilities are an example


# ---------------------------------------------------------------------- 113 (local)
def test_local_measurements_have_the_right_shape_and_ranges():
    loc = S.local_data()
    n = S.N_BYTES
    assert loc["deltas"].shape == loc["clock_high"].shape == loc["os_bytes"].shape == (n,)
    assert (loc["deltas"] >= 0).all() and loc["os_bytes"].dtype == np.uint8
    for b in S.local_sources():
        assert b.dtype == np.uint8 and b.shape == (n,)
    s = study("entropy-sources")
    assert s.metadata["local"] is True and "differ" in s.metadata["evidence"]["local"]
    m = np.asarray(s.panels[0].data["values"])                  # (source, estimate)
    _jitter, system, counter, clock = m
    assert m.shape == (4, 3) and (m >= 0).all() and (m <= 8).all()
    assert (m[:, 1] <= m[:, 0] + 1e-9).all()                    # min-entropy never above Shannon
    # what must hold on any machine
    assert counter[0] == 8.0 and counter[2] < 1e-6              # perfect counts, certain order
    assert system[0] > 7.99 and system[2] > 7.5
    assert clock[0] <= 1.0 and clock[0] < system[0] - 6
    v = np.asarray(s.panels[1].data["local_timer_bit"])
    assert v.shape == (5,) and 0 <= v[0] <= 1 and 0 <= v[1] <= 1 and 0 <= v[2] <= 0.5


def test_sources_claims():
    s = study("entropy-sources")
    vn = np.asarray(s.panels[1].data["all_seeds"])              # (seed, before/after/kept/..)
    assert np.abs(vn[:, 0] - S.BIAS).max() < 0.005 and np.abs(vn[:, 1] - 0.5).max() < 0.01
    assert np.abs(vn[:, 2] - S.BIAS * (1 - S.BIAS)).max() < 0.005
    coin = np.asarray(s.panels[1].data["coin_entropy"])
    assert (coin[:, 1] > coin[:, 0] + 0.25).all() and coin[:, 1].min() > 0.999
    measured, exact = np.asarray(s.panels[2].data["ys"])
    assert np.abs(measured - exact).max() < 0.005
    assert abs(measured[-1] - 0.5) < abs(measured[0] - 0.5) / 5
    assert float(s.panels[2].data["same_stream"]) == 0.0        # a source XORed with itself
    assert abs(s.panels[2].data["hash_bits_changed"][0] - 128) < 2
    assert [int(v) for v in s.panels[3].data["values"]] == [2 ** k for k in S.STRETCH_BITS]
    stretched = np.asarray(s.panels[3].data["stretched_estimates"])
    assert stretched[0] > 7.99 and stretched[4] >= 8.0          # looks perfect; holds 16 bits


def test_sources_data_is_what_it_says():
    d = S.sources_data()
    demo = S.biased_bits(S.VN_SEEDS[0])[:32]
    assert np.array_equal(d["vn_in"], demo) and np.array_equal(d["vn_out"], S.von_neumann(demo))
    row = S.cleaning(S.biased_bits(1))
    assert np.allclose(row, d["vn"][1])
    assert np.isclose(d["xor_exact"][1], 2 * S.BIAS * (1 - S.BIAS))
    small = S.pcg32_seeded(12345, 1024).astype("<u4").view(np.uint8)
    assert np.array_equal(d["stretched_bytes"], small)
    pool, seed = S.pool_seed()
    assert len(pool) == 128 and len(seed) == 32
    assert pool[64:96] == bytes(range(32))                      # the counter's part of the pool


def test_sources_numbers_on_screen():
    s = study("entropy-sources")
    anim = M.EntropySources(s)
    sub = subtitles(anim)
    m = np.asarray(s.panels[0].data["values"])
    vn = np.asarray(s.panels[1].data["values"])
    assert "scores %.1f bits per byte, the system's bytes %.2f" % (m[0, 0], m[1, 0]) in sub[1]
    assert "from %.0f%% to %.1f%%, and %.2f bits" % (100 * vn[0], 100 * vn[1], vn[2]) in sub[2]
    assert "exactly 1,048,576 different streams" in sub[4]
    assert "%.2f bits per byte by the usual estimate" % s.panels[3].data[
        "stretched_estimates"][0] in sub[0]
    shown = texts(anim, mid(anim, 1))
    assert M.LOCAL in shown                                     # labelled as this machine's
    assert {"%.2f" % v for v in m.ravel()} <= set(shown)
    assert any(s.metadata["machine"] in t for t in shown)


# ---------------------------------------------------------------------- 114
@needs_text
def test_entropy_claims():
    s = study("measuring-entropy")
    m = np.asarray(s.panels[0].data["values"])                  # (source, estimate)
    ideal = np.asarray(s.panels[0].data["ideal"])
    text, _packed, gen, bad, system, counter = range(6)
    single, least, given1, given2, squeezed = range(5)
    assert (m[:, least] <= m[:, single] + 1e-9).all()           # min-entropy never above Shannon
    assert m[text, single] < 5.5 and m[1:, single].min() > 7.99
    # structure that single bytes miss
    assert m[text, given1] < m[text, single] - 0.5 and m[text, given2] < m[text, given1] - 0.3
    for row in (bad, counter):
        assert m[row, single] == 8.0 and m[row, given1] < 1e-6 and m[row, squeezed] < 0.2
    assert m[text, squeezed] < 4.0 and m[text, squeezed] < m[text, single] - 1.0
    # the generator is indistinguishable from an ideal source on every estimate
    assert np.abs(m[gen] - ideal).max() < 0.03 and m[gen, squeezed] >= 8.0
    assert np.abs(m[system, [single, given1]] - ideal[[single, given1]]).max() < 0.05
    # too few bytes for triples: even an ideal source scores far below 8
    assert ideal[given2] < 3.0 and ideal[given1] > 7.7
    tests = np.asarray(s.panels[2].data["values"])
    assert (np.abs(s.panels[2].data["generator_seeds"]) < 3).all()      # passes, three seeds
    for row in (text, bad, counter):
        assert (np.abs(tests[row]) > 3).all()
    assert tests[bad, 0] < -3 and tests[counter, 0] < -3        # too even to be chance
    assert np.isfinite(tests).all()
    # the plug-in estimate is biased low, and less so with more samples
    est, true = np.asarray(s.panels[1].data["ys"]), np.asarray(s.panels[1].data["true"])
    assert est[0, 0] <= 4.0 and (np.diff(est[0]) > 0).all() and est[0, -1] < 8.0
    assert (est[0] < true[0]).all() and (est[1, :8] < true[1]).all()
    assert abs(est[0, -1] - 8) < 0.001 and abs(est[1, -1] - true[1]) < 0.001
    # 32 bits of seed, whatever the bytes score
    assert [int(v) for v in s.panels[3].data["values"]] == [2 ** 32, 2 ** 16, 1, 1]
    assert [int(v) for v in s.panels[3].data["recovered"]] == [S.SECRET]
    assert s.metadata["generator"]["true_entropy_bits"] == 32.0


@needs_text
def test_entropy_data_is_what_it_says():
    from voxanim import realdata

    d = S.entropy_data()
    srcs = S.entropy_sources_fixed()
    raw = realdata.text("train").encode("utf-8")
    assert srcs[0].tobytes() == raw[:S.N_BYTES]
    assert zlib.decompress(zlib.compress(raw, 9)) == raw
    assert srcs[1].tobytes() == zlib.compress(raw, 9)[2:S.N_BYTES + 2]
    assert np.array_equal(srcs[2][:6], [v for out in S.pcg16(S.SECRET, 3)
                                        for v in (out >> 8, out & 255)])
    assert np.array_equal(srcs[3][:4], [S.lcg_states(1, 4, *S.LCG32)[k] & 255 for k in range(4)])
    assert srcs[S.ENT_OS] is None
    assert np.array_equal(d["strips"][S.ENT_OS], S.local_data()["os_bytes"][:4096])   # local row
    assert d["measures"].shape == (6, 5) and d["tests"].shape == (6, 3)
    assert d["pairs"].shape == (6, 256, 256) and (d["pairs"].sum((1, 2)) == S.N_BYTES - 1).all()
    assert np.array_equal(d["hist"][0], np.bincount(srcs[0], minlength=256))
    assert np.allclose(d["measures"][0], S.byte_measures(srcs[0]))
    assert np.array_equal(d["shown"], S.pcg16(S.SECRET, S.SHOWN))
    # the search tries every seed of a block: the same seeds as computing each one's output
    target = int(d["shown"][0])
    for start in (0, 11 << 28):
        seeds = np.arange(start, start + (1 << 22), dtype=np.uint64).astype(np.uint32)
        assert np.array_equal(S.pcg16_first_matches(start, start + (1 << 22), target),
                              seeds[S.pcg16_output(seeds) == target])
    state = np.array([S.SECRET], np.uint32)
    assert int(S.pcg16_output(state)[0]) == target


@needs_text
def test_entropy_numbers_on_screen():
    s = study("measuring-entropy")
    anim = M.MeasuringEntropy(s)
    sub = subtitles(anim)
    m = np.asarray(s.panels[0].data["values"])
    assert "Text scores %.2f bits per byte" % m[0, 0] in sub[1]
    assert "the text falls to %.2f bits" % m[0, 2] in sub[3]
    assert "%.2f for the text" % m[0, 4] in sub[3]
    assert "leaves 65,536 after one and 1 after two" in sub[5]
    assert "%.1f, %.1f and %.1f" % tuple(s.panels[2].data["values"][2]) in sub[4]
    shown = texts(anim, mid(anim, 3))
    for k in range(6):
        for col in (0, 2, 3, 4):
            assert "%.2f" % m[k, col] in shown
    assert "the seed: %s, found by trying every one of 2³²" % M.commas(S.SECRET) in texts(
        anim, mid(anim, 5))
    assert M.commas(2 ** 32) in texts(anim, mid(anim, 5))

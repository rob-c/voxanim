"""The experiments behind animations 110-114 (see :func:`voxanim.next_catalog.redesigned`):
pseudo-random generators, seeds, sampling, entropy sources and measuring entropy.

Four kinds of evidence are kept apart here and labelled on screen:

* **exact arithmetic**: a generator's recurrence, a period (proved by jumping the
  recurrence ahead in integer arithmetic), a count over every seed;
* **statistics of a generator's output**: deterministic given the seed;
* **real text**: Tiny Shakespeare (:mod:`voxanim.realdata`), in ``measuring-entropy`` only;
* **local measurements** of the machine that built the cache (the operating system's
  random bytes, the jitter of a timer): measured once by :func:`local_data`, cached, and
  different on every machine. No test asserts their values.

The small generators are written out in integer arithmetic (a linear congruential
generator, RANDU, xorshift32, PCG's output function, splitmix64) so every number an
animation shows can be followed; numpy's generator is the well-behaved reference.
Long streams are produced by *doubling*: the map that advances a generator ``k`` steps
is composed with itself, so a million consecutive states take twenty array operations
and are exactly the states the one-step recurrence gives (the tests compare them).
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
import zlib

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
M32 = (1 << 32) - 1
M64 = (1 << 64) - 1

TINY = (5, 3, 4)                              # x -> 5x + 3 mod 16: every state once
TINY_BAD = (5, 4, 4)                          # the same with + 4: short cycles
LCG32 = (1664525, 1013904223, 32)             # "Numerical Recipes" constants, mod 2^32
RANDU = (65539, 0, 31)                        # IBM's RANDU, mod 2^31, odd seeds
PCG_MULT, PCG_INC = 6364136223846793005, 1442695040888963407       # PCG32's 64-bit state
PCG16_MULT, PCG16_INC = 747796405, 2891336453                      # PCG 32/16's 32-bit state
XS_SHIFTS = (13, 17, 5)                       # Marsaglia's xorshift32
N_BYTES = 1 << 18                             # bytes of every source whose entropy is estimated


def _settings(**more):
    return dict(version=VERSION, **more)


# ====================================================================== generators
def lcg_step(x, a, c, bits):
    """One step of ``x -> a x + c mod 2^bits`` (Python integers: exact)."""
    return (a * x + c) & ((1 << bits) - 1)


def lcg_power(a, c, bits, steps):
    """``(A, C)`` with ``x -> A x + C`` equal to ``steps`` steps of the generator."""
    mask = (1 << bits) - 1
    ra, rc = 1, 0
    while steps:
        if steps & 1:
            ra, rc = (a * ra) & mask, (a * rc + c) & mask
        a, c = (a * a) & mask, (a * c + c) & mask
        steps >>= 1
    return ra, rc


def lcg_states(seed, n, a, c, bits):
    """The ``n`` states after ``seed`` (``x1 .. xn``) as uint64, by doubling."""
    mask = (1 << bits) - 1
    out = np.empty(n, np.uint64)
    out[0] = (a * seed + c) & mask
    have = 1
    while have < n:
        k = min(have, n - have)
        out[have:have + k] = (out[:k] * np.uint64(a) + np.uint64(c)) & np.uint64(mask)
        a, c = (a * a) & mask, (a * c + c) & mask
        have += k
    return out


def lcg_cycle(seed, a, c, bits):
    """The states from ``seed`` until it comes round again (the seed first)."""
    out, x = [seed], lcg_step(seed, a, c, bits)
    while x != seed:
        out.append(x)
        x = lcg_step(x, a, c, bits)
    return out


def power_of_two_period(a, c, bits, seed, log2_period):
    """True when the state ``seed`` returns after exactly ``2^log2_period`` steps: it is
    back after that many and not after half as many, and a period that divides a power
    of two is one."""
    def at(steps):
        ra, rc = lcg_power(a, c, bits, steps)
        return (ra * seed + rc) & ((1 << bits) - 1)
    return at(1 << log2_period) == seed and at(1 << (log2_period - 1)) != seed


def xs_step(x):
    """One step of xorshift32: three shifts, each XORed back into the state."""
    a, b, c = XS_SHIFTS
    x ^= (x << a) & M32
    x ^= x >> b
    x ^= (x << c) & M32
    return x


def _xs_apply(cols, x):
    out, b = 0, 0
    while x:
        if x & 1:
            out ^= cols[b]
        x >>= 1
        b += 1
    return out


def xs_power(steps):
    """``steps`` steps of xorshift32 as the images of the 32 single-bit states (the
    step is linear over bits, so these say where every state goes)."""
    base = [xs_step(1 << b) for b in range(32)]
    result = [1 << b for b in range(32)]
    while steps:
        if steps & 1:
            result = [_xs_apply(base, col) for col in result]
        base = [_xs_apply(base, col) for col in base]
        steps >>= 1
    return result


def xs_jump(x, steps):
    """The xorshift32 state ``steps`` steps after ``x``."""
    return _xs_apply(xs_power(steps), x)


def _xs_apply_array(cols, arr):
    out = np.zeros_like(arr)
    for b in range(32):
        out ^= ((arr >> np.uint64(b)) & np.uint64(1)) * np.uint64(cols[b])
    return out


def xs_states(seed, n):
    """The ``n`` xorshift32 states after ``seed``, by doubling."""
    out = np.empty(n, np.uint64)
    out[0] = xs_step(seed)
    cols = [xs_step(1 << b) for b in range(32)]
    have = 1
    while have < n:
        k = min(have, n - have)
        out[have:have + k] = _xs_apply_array(cols, out[:k])
        cols = [_xs_apply(cols, col) for col in cols]
        have += k
    return out


def splitmix64(x):
    """The splitmix64 hash of a 64-bit integer: add a constant, then two rounds of
    xor-shift and multiply, and a last xor-shift."""
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def splitmix64_stages(x):
    """The value after each of splitmix64's seven steps (the last is :func:`splitmix64`):
    the input, add, xor-shift, multiply, xor-shift, multiply, xor-shift."""
    z0 = (x + 0x9E3779B97F4A7C15) & M64
    z1 = z0 ^ (z0 >> 30)
    z2 = (z1 * 0xBF58476D1CE4E5B9) & M64
    z3 = z2 ^ (z2 >> 27)
    z4 = (z3 * 0x94D049BB133111EB) & M64
    return [x, z0, z1, z2, z3, z4, z4 ^ (z4 >> 31)]


def splitmix64_array(x):
    """:func:`splitmix64` of every element of a uint64 array."""
    z = np.asarray(x, np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def pcg32_parts(old):
    """PCG32's output function on a 64-bit state: ``(xorshifted, rot, output)``. The top
    five bits say how far to rotate 32 bits taken from the middle and top."""
    old = np.asarray(old, np.uint64)
    xorshifted = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)) & np.uint64(M32)
    rot = old >> np.uint64(59)
    out = ((xorshifted >> rot) | (xorshifted << ((np.uint64(32) - rot) & np.uint64(31))))
    return xorshifted, rot, out & np.uint64(M32)


def pcg32(state, n, inc=PCG_INC):
    """``n`` outputs of PCG32 (XSH-RR 64/32) from the 64-bit ``state``: each is read
    from the state, which then takes one step of a 64-bit linear congruential generator."""
    old = np.empty(n, np.uint64)
    old[0] = state
    if n > 1:
        old[1:] = lcg_states(state, n - 1, PCG_MULT, inc, 64)
    return pcg32_parts(old)[2]


def pcg32_reference(initstate, initseq, n):
    """PCG32 seeded as its reference implementation seeds it (``pcg32_srandom_r``)."""
    inc = ((initseq << 1) | 1) & M64
    s = lcg_step(0, PCG_MULT, inc, 64)
    s = lcg_step((s + initstate) & M64, PCG_MULT, inc, 64)
    return pcg32(s, n, inc)


def pcg32_seeded(seed, n):
    """PCG32 with its state set to the splitmix64 hash of ``seed``."""
    return pcg32(splitmix64(seed), n)


def pcg16_output(state):
    """PCG XSH-RR 32/16: 16 bits out of a 32-bit state (uint32 array in, uint32 out)."""
    s = np.asarray(state, np.uint32)
    x = (((s >> np.uint32(10)) ^ s) >> np.uint32(12)) & np.uint32(0xFFFF)
    r = s >> np.uint32(28)
    return ((x >> r) | (x << ((np.uint32(16) - r) & np.uint32(15)))) & np.uint32(0xFFFF)


def pcg16(seed, n):
    """``n`` 16-bit outputs of the 32-bit-state PCG whose state starts at ``seed``."""
    old = np.empty(n, np.uint64)
    old[0] = seed
    if n > 1:
        old[1:] = lcg_states(seed, n - 1, PCG16_MULT, PCG16_INC, 32)
    return pcg16_output(old.astype(np.uint32))


def pcg16_bytes(seed, n):
    """``n`` bytes from :func:`pcg16`: each output's high byte, then its low byte."""
    out = pcg16(seed, (n + 1) // 2)
    return np.stack([out >> np.uint32(8), out & np.uint32(255)], 1).ravel()[:n].astype(np.uint8)


def unit(x, bits=32):
    """Integers below ``2^bits`` as numbers in [0, 1)."""
    return np.asarray(x, np.float64) / float(1 << bits)


def stream(kind, seed, n):
    """``n`` outputs in [0, 1) of generator ``kind`` and their integers (``(u, x, bits)``)."""
    if kind == "randu":
        x = lcg_states(seed, n, *RANDU)
        return unit(x, 31), x, 31
    if kind == "lcg32":
        x = lcg_states(seed, n, *LCG32)
    elif kind == "xorshift32":
        x = xs_states(seed, n)
    elif kind == "pcg32":
        x = pcg32_seeded(seed, n)
    elif kind == "numpy":
        x = np.random.default_rng(seed).integers(0, 1 << 32, n, dtype=np.uint64)
    else:
        raise ValueError(kind)
    return unit(x), x, 32


# ====================================================================== statistics
def serial_correlation(u):
    """Pearson correlation of each value with the next."""
    u = np.asarray(u, np.float64)
    return float(np.corrcoef(u[:-1], u[1:])[0, 1])


def cells_hit(u, side):
    """Fraction of the ``side^3`` cells of the unit cube that the consecutive triples
    ``(u[k], u[k+1], u[k+2])`` fall in."""
    q = np.minimum((np.asarray(u) * side).astype(np.int64), side - 1)
    idx = (q[:-2] * side + q[1:-1]) * side + q[2:]
    return float(np.count_nonzero(np.bincount(idx, minlength=side ** 3))) / side ** 3


def sequence_period(seq, candidates):
    """The smallest of ``candidates`` that is a period of ``seq`` (0: none)."""
    seq = np.asarray(seq)
    for p in sorted(candidates):
        if 2 * p <= len(seq) and np.array_equal(seq[p:], seq[:-p]):
            return int(p)
    return 0


def short_period(bits_seq, limit=65536):
    """The smallest period up to ``limit`` of a 0/1 sequence, or 0 when there is none: a
    period ``p`` needs the 64 values from ``p`` on to repeat the first 64, which is
    checked for every ``p``; a candidate is then checked along the whole sequence."""
    b = np.asarray(bits_seq, np.uint64)
    n = len(b) - 64
    window = np.zeros(n, np.uint64)
    for j in range(64):
        window |= b[j:n + j] << np.uint64(j)
    for p in np.nonzero(window[1:limit + 1] == window[0])[0] + 1:
        if np.array_equal(b[p:], b[:-p]):
            return int(p)
    return 0


def shannon(counts):
    """Shannon entropy, in bits, of the distribution given by ``counts``."""
    c = np.asarray(counts, np.float64).ravel()
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum()) + 0.0          # + 0.0: never "-0.0"


def block_counts(b, k):
    """How often each run of ``k`` consecutive bytes occurs."""
    b = np.asarray(b, np.int64)
    if k == 1:
        return np.bincount(b, minlength=256)
    idx = b[:len(b) - k + 1].copy()
    for j in range(1, k):
        idx = idx * 256 + b[j:len(b) - k + 1 + j]
    if k == 2:
        return np.bincount(idx, minlength=65536)
    return np.unique(idx, return_counts=True)[1]


def byte_entropy(b):
    """The plug-in estimate of the Shannon entropy of single bytes, bits per byte."""
    return shannon(block_counts(b, 1))


def conditional_entropy(b, given):
    """The plug-in estimate of the Shannon entropy of a byte given the ``given`` bytes
    before it: the entropy of blocks of ``given + 1`` minus that of blocks of ``given``."""
    return max(0.0, shannon(block_counts(b, given + 1)) - shannon(block_counts(b, given)))


def min_entropy(b):
    """The min-entropy estimate, bits per byte: minus log2 of the commonest byte's share."""
    c = block_counts(b, 1)
    return float(-np.log2(c.max() / c.sum())) + 0.0


def zlib_bits(b):
    """Bits per byte after zlib (level 9): an upper bound on the entropy rate."""
    raw = np.asarray(b, np.uint8).tobytes()
    return 8.0 * len(zlib.compress(raw, 9)) / len(raw)


def chi_square_z(b):
    """The chi-square statistic of the 256 byte counts against equal shares, as a
    z-score (255 degrees of freedom: mean 255, variance 510). Large: uneven; very
    negative: more even than chance allows."""
    c = block_counts(b, 1).astype(np.float64)
    e = c.sum() / 256.0
    return float((((c - e) ** 2 / e).sum() - 255.0) / np.sqrt(510.0))


def serial_z(b):
    """The correlation of each byte with the next, times sqrt(n): a z-score."""
    b = np.asarray(b, np.float64)
    return float(np.corrcoef(b[:-1], b[1:])[0, 1] * np.sqrt(len(b) - 1))


def runs_z(b):
    """The Wald-Wolfowitz runs test on bytes above or below their median, as a z-score:
    negative when values come in longer runs than chance gives."""
    above = np.asarray(b) > np.median(b)
    n = len(above)
    n1 = int(above.sum())
    n2 = n - n1
    runs = 1 + int(np.count_nonzero(above[1:] != above[:-1]))
    mean = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    return float(np.clip((runs - mean) / np.sqrt(max(var, 1e-12)), -999.0, 999.0))


def coin_entropy(p):
    """Shannon entropy, in bits, of a coin that shows 1 with probability ``p``."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def von_neumann(bits_in):
    """Von Neumann's trick: read the bits in pairs; 01 gives 0, 10 gives 1, 00 and 11
    give nothing."""
    b = np.asarray(bits_in, np.uint8)
    pairs = b[:len(b) // 2 * 2].reshape(-1, 2)
    keep = pairs[:, 0] != pairs[:, 1]
    return pairs[keep, 0]


# ====================================================================== 110: generators
PRG_KINDS = ("randu", "lcg32", "xorshift32", "pcg32", "numpy")
PRG_NAMES = ("RANDU", "32-bit LCG", "xorshift32", "PCG32", "numpy")
PRG_SEEDS = (1, 3, 5)                         # odd: RANDU needs an odd seed
PRG_N = 1 << 20
PRG_SIDE = 64                                 # the cube is cut into 64^3 cells


def _prg_build():
    out = {}
    out["tiny"] = np.array(lcg_cycle(1, *TINY) + [1], np.int64)
    out["tiny_bad"] = np.array(lcg_cycle(1, *TINY_BAD) + [1], np.int64)
    out["lcg_rows"] = lcg_states(1, 16, *LCG32)
    x = lcg_states(1, 1 << 18, *LCG32)
    out["bit_period"] = np.array([
        sequence_period((x >> np.uint64(k)) & np.uint64(1),
                        [1 << j for j in range(k + 2)]) for k in range(16)], np.int64)
    x = lcg_states(1, PRG_N + 2, *RANDU).astype(np.int64)
    combo = 9 * x[:-2] - 6 * x[1:-1] + x[2:]
    planes = combo >> 31
    assert np.array_equal(planes << 31, combo)
    out["randu_planes"] = np.unique(planes)
    out["randu_points"] = unit(x[:6002], 31).astype(np.float32)
    out["good_points"] = stream("pcg32", 1, 6002)[0].astype(np.float32)
    stats = np.zeros((len(PRG_KINDS), len(PRG_SEEDS), 4))
    for g, kind in enumerate(PRG_KINDS):
        for s, seed in enumerate(PRG_SEEDS):
            u, xi, _bits = stream(kind, seed, PRG_N + 2)
            low = (xi & np.uint64(1)).astype(np.uint8)
            stats[g, s] = (serial_correlation(u), cells_hit(u, PRG_SIDE),
                           short_period(low[:1 << 18]),
                           float(np.mean(low[1:] != low[:-1])))
    out["stats"] = stats
    # one xorshift step and one PCG output, on real states (seed 1, eight steps on)
    x0 = int(xs_states(1, 8)[-1])
    a, b, c = XS_SHIFTS
    t1 = (x0 << a) & M32
    x1 = x0 ^ t1
    t2 = x1 >> b
    x2 = x1 ^ t2
    t3 = (x2 << c) & M32
    out["xs_demo"] = np.array([x0, t1, x1, t2, x2, t3, x2 ^ t3], np.uint64)
    old = lcg_states(splitmix64(1), 8, PCG_MULT, PCG_INC, 64)[-1:]
    xorshifted, rot, word = pcg32_parts(old)
    out["pcg_demo"] = np.array([old[0], xorshifted[0], rot[0], word[0]], np.uint64)
    return out


def prg_data():
    """Everything ``pseudo-random-generators`` draws (exact, or statistics of outputs)."""
    return cached_npz("random-generators", _settings(n=PRG_N, side=PRG_SIDE, seeds=PRG_SEEDS),
                      _prg_build)


def prg_periods():
    """The periods, each proved in integer arithmetic: ``{name: (text, log2, proved)}``."""
    order = next(k for k in range(1, 32) if pow(RANDU[0], 1 << k, 1 << 31) == 1)
    full = (1 << 32) - 1
    xs_ok = xs_jump(1, full) == 1 and all(xs_jump(1, full // p) != 1
                                           for p in (3, 5, 17, 257, 65537))
    return {
        "tiny": ("16", 4.0, len(lcg_cycle(1, *TINY)) == 16),
        "randu": ("2^%d" % order, float(order), pow(RANDU[0], 1 << (order - 1), 1 << 31) != 1),
        "lcg32": ("2^32", 32.0, power_of_two_period(*LCG32, 1, 32)),
        "xorshift32": ("2^32 - 1", float(np.log2(full)), bool(xs_ok)),
        "pcg32": ("2^64", 64.0, power_of_two_period(PCG_MULT, PCG_INC, 64, splitmix64(1), 64)),
    }


def generators():
    d = prg_data()
    periods = prg_periods()
    stats = d["stats"]
    mean = stats.mean(1)
    ideal = 1.0 - np.exp(-PRG_N / PRG_SIDE ** 3)
    pts = d["randu_points"]
    view = np.stack([pts[:-2], 9 * pts[:-2] - 6 * pts[1:-1] + pts[2:]], 1)
    panels = (
        Panel("A 4-bit generator's whole cycle",
              "x -> 5x + 3 mod 16 from 1: every state once, then 1 again (exact).", "curve",
              dict(x=np.arange(len(d["tiny"])), ys=d["tiny"][None].astype(float),
                   labels=np.array(["state"]), short_cycle=d["tiny_bad"])),
        Panel("Periods, proved by jumping the recurrence ahead",
              "log2 of the number of steps before each generator repeats (exact).", "bars",
              dict(labels=np.array(list(periods)),
                   values=np.array([v[1] for v in periods.values()]),
                   low_bit_period=d["bit_period"])),
        Panel("What separates bad from good",
              "Rows: RANDU, 32-bit LCG, xorshift32, PCG32, numpy. Columns: serial "
              "correlation, share of 64^3 cells hit by 2^20 triples, lowest-bit period "
              "(0: none up to 65536), how often the lowest bit changes.", "matrix",
              dict(values=mean, all_seeds=stats, ideal_cells=np.array(ideal))),
        Panel("RANDU's triples lie on planes",
              "Each consecutive triple (x, y, z): x across, 9x - 6y + z up. Only %d "
              "values occur." % len(d["randu_planes"]), "points",
              dict(xy=view, planes=d["randu_planes"])),
    )
    return Study(
        "How a computer makes random numbers",
        "A small state, scrambled and read, again and again",
        panels,
        metadata=dict(
            evidence=dict(exact="recurrences, cycles and periods", statistics="the table: "
                          "outputs of each generator from seeds %s" % (PRG_SEEDS,)),
            generators=dict(
                tiny="x -> 5x + 3 mod 16", lcg32="x -> 1664525 x + 1013904223 mod 2^32",
                randu="x -> 65539 x mod 2^31", xorshift32="shifts 13, 17, 5",
                pcg32="64-bit LCG state, XSH-RR output, state = splitmix64(seed)",
                numpy="numpy.random.default_rng(seed): PCG64"),
            seeds=list(PRG_SEEDS), outputs_per_seed=PRG_N + 2,
            periods={k: dict(period=v[0], proved=bool(v[2])) for k, v in periods.items()},
            period_method="a power-of-two period p: the state is back after p steps and "
                          "not after p/2 (affine maps composed by repeated squaring); "
                          "RANDU: the order of 65539 mod 2^31; xorshift32: back after "
                          "2^32 - 1 steps and not after (2^32 - 1)/q for each prime q",
            low_bit_period_of_lcg32=d["bit_period"].tolist(),
            randu_planes=int(len(d["randu_planes"])),
            cells=dict(side=PRG_SIDE, triples=PRG_N, ideal_share=float(ideal),
                       share={n: stats[g, :, 1].tolist() for g, n in enumerate(PRG_NAMES)}),
            serial_correlation={n: stats[g, :, 0].tolist() for g, n in enumerate(PRG_NAMES)},
            limitations="Three simple statistics, not a test battery: xorshift32 passes "
                        "all three here and still fails stronger tests (it is linear "
                        "over bits). Serial correlation is near zero for every "
                        "generator, RANDU included: it cannot see RANDU's planes.",
            designs_tried=["as built: the generators, statistics, seeds and sample sizes "
                           "were fixed before the table was computed"],
        ),
    )


# ====================================================================== 111: seeds
SEED_N = 100_000
SEED_FIRST = 256                              # first outputs of seeds 1 .. 256
SEED_KINDS = ("lcg32", "pcg32", "numpy")
SEED_NAMES = ("LCG, seed used as the state", "PCG32, seed hashed", "numpy")
SEED_PAIRS = ((0, 1), (0, 2), (1, 2))         # of the streams from seeds 1, 2, 3
PAR_TRIALS, PAR_WORKERS, PAR_DARTS = 400, 4, 1000
PAR_JUMP = 1 << 40                            # steps between the jumped streams


def first_outputs(kind, seeds):
    """The first output in [0, 1) of generator ``kind`` for each seed."""
    seeds = np.asarray(seeds, np.uint64)
    if kind == "lcg32":
        return unit((seeds * np.uint64(LCG32[0]) + np.uint64(LCG32[1])) & np.uint64(M32))
    if kind == "pcg32":
        return unit(pcg32_parts(splitmix64_array(seeds))[2])
    return np.array([np.random.default_rng(int(s)).integers(0, 1 << 32, dtype=np.uint64)
                     for s in seeds]) / float(1 << 32)


def pi_estimate(u):
    """4 x the share of the points ``(u[2k], u[2k+1])`` inside the quarter circle."""
    p = np.asarray(u).reshape(-1, 2)
    return 4.0 * float(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2 < 1.0))


def _seeds_build():
    out = {}
    a, again, other = (pcg32_seeded(s, SEED_N) for s in (42, 42, 43))
    out["same"] = np.stack([a[:8], again[:8], other[:8]])
    agree = 1.0 - np.unpackbits((a ^ other).astype(">u4").view(np.uint8)).mean()
    ref = [np.random.default_rng(s).integers(0, 1 << 32, SEED_N, dtype=np.uint64)
           for s in (42, 42, 43)]
    out["equal"] = np.array([np.mean(a == again), np.mean(a == other), agree,
                             np.mean(ref[0] == ref[1]), np.mean(ref[0] == ref[2])])
    seeds = np.arange(1, SEED_FIRST + 1)
    first = np.stack([first_outputs(k, seeds) for k in SEED_KINDS])
    out["first"] = first
    out["first_corr"] = np.array([np.corrcoef(seeds, f)[0, 1] for f in first])
    out["first_gap"] = np.abs(np.diff(first, axis=1)).mean(1)
    streams = [np.stack([stream(k, s, SEED_N)[1] for s in (1, 2, 3)]) for k in SEED_KINDS]
    out["streams"] = np.stack([s[:, :8] for s in streams])
    # the third stream computed from the other two, and the usual correlation
    out["relation"] = np.array([
        np.mean(((np.uint64(2) * s[1] - s[0]) & np.uint64(M32)) == s[2]) for s in streams])
    out["pearson"] = np.array([[np.corrcoef(unit(s[i]), unit(s[j]))[0, 1]
                                for i, j in SEED_PAIRS] for s in streams])
    bit = np.arange(32, dtype=np.uint64)
    one = np.uint64(1)
    out["bit_agree"] = np.array([[
        np.mean(((s[i][:, None] >> bit) & one) == ((s[j][:, None] >> bit) & one), axis=0)
        for i, j in SEED_PAIRS] for s in streams])
    # hashing a seed: how many of the 64 bits change when one bit of the seed does
    base = np.arange(1, 1001, dtype=np.uint64)
    hashed = splitmix64_array(base)
    flips = np.array([np.unpackbits((hashed ^ splitmix64_array(base ^ np.uint64(1 << b)))
                                    .view(np.uint8)).reshape(len(base), 64).sum(1)
                      for b in range(64)])
    out["avalanche"] = np.array([flips.mean(), flips.min(), flips.max()])
    out["hashed"] = np.array([splitmix64(s) for s in (1, 2, 3)], np.uint64)
    # four workers estimating pi: one seed reused, one stream jumped ahead, spawned streams
    est = np.zeros((3, PAR_TRIALS, PAR_WORKERS))
    corr = np.zeros((3, PAR_TRIALS))
    need = 2 * PAR_DARTS
    jumps = [lcg_power(PCG_MULT, PCG_INC, 64, w * PAR_JUMP) for w in range(PAR_WORKERS)]
    for t in range(PAR_TRIALS):
        state = splitmix64(1000 + t)
        spawned = np.random.SeedSequence(1000 + t).spawn(PAR_WORKERS)
        us = (
            [unit(pcg32(state, need))] * PAR_WORKERS,
            [unit(pcg32((ja * state + jc) & M64, need)) for ja, jc in jumps],
            [np.random.default_rng(s).random(need) for s in spawned],
        )
        for m, workers in enumerate(us):
            est[m, t] = [pi_estimate(u) for u in workers]
            corr[m, t] = np.corrcoef(workers[0], workers[1])[0, 1]
    out["pi"] = est
    out["worker_corr"] = corr.mean(1)
    state = splitmix64(1000)
    out["darts"] = np.stack([unit(pcg32((ja * state + jc) & M64, need)).reshape(-1, 2)
                             for ja, jc in jumps]).astype(np.float32)
    return out


def seeds_data():
    """Everything ``seeds-and-streams`` draws (statistics of generators' outputs)."""
    return cached_npz("random-seeds", _settings(n=SEED_N, first=SEED_FIRST, trials=PAR_TRIALS,
                                                workers=PAR_WORKERS, darts=PAR_DARTS),
                      _seeds_build)


def parallel_errors(d):
    """Root-mean-square error of the pi estimates over the trials: ``(one worker, the
    mean of the four workers)`` for each way of seeding."""
    est = d["pi"]
    one = np.sqrt(((est[:, :, 0] - np.pi) ** 2).mean(1))
    four = np.sqrt(((est.mean(2) - np.pi) ** 2).mean(1))
    return one, four


def seeds():
    d = seeds_data()
    one, four = parallel_errors(d)
    names = SEED_NAMES
    panels = (
        Panel("Same seed, same stream",
              "Share of %d outputs that are equal: seed 42 twice, then 42 against 43 "
              "(own PCG32; numpy's generator)." % SEED_N, "bars",
              dict(labels=np.array(["42 twice", "42 and 43", "bits agreeing, 42 and 43",
                                    "numpy 42 twice", "numpy 42 and 43"]),
                   values=d["equal"])),
        Panel("First output against seed",
              "Seeds 1 to %d: the LCG's first outputs lie on a line." % SEED_FIRST, "points",
              dict(xy=np.stack([np.arange(1, SEED_FIRST + 1).astype(float), d["first"][0]], 1),
                   hashed=d["first"][1], numpy=d["first"][2], correlation=d["first_corr"],
                   adjacent_gap=d["first_gap"])),
        Panel("Streams from seeds 1, 2 and 3, bit by bit",
              "Share of outputs whose bit k agrees, for the LCG: rows are the pairs "
              "(1, 2), (1, 3), (2, 3); columns the bits, lowest first.", "matrix",
              dict(values=d["bit_agree"][0], hashed=d["bit_agree"][1],
                   numpy=d["bit_agree"][2], relation=d["relation"], pearson=d["pearson"])),
        Panel("Four workers estimate pi",
              "Root-mean-square error of the mean of four workers' estimates over %d "
              "trials: one seed reused, one stream jumped ahead, spawned streams; then "
              "one worker alone." % PAR_TRIALS, "bars",
              dict(labels=np.array(["same seed", "jumped", "spawned", "one worker"]),
                   values=np.append(four, one.mean()), one_worker=one,
                   worker_correlation=d["worker_corr"])),
    )
    return Study(
        "Seeds and streams",
        "What a seed fixes, and how streams should differ",
        panels,
        metadata=dict(
            evidence="statistics of generators' outputs: deterministic given the seeds",
            generators=dict(zip(("weak", "own", "reference"), names)),
            seeds=dict(same=[42, 42, 43], streams=[1, 2, 3], first_outputs=[1, SEED_FIRST],
                       parallel_trials=[1000, 1000 + PAR_TRIALS - 1]),
            outputs_compared=SEED_N,
            first_output_correlation_with_seed=dict(zip(names, d["first_corr"].tolist())),
            mean_gap_between_adjacent_seeds=dict(zip(names, d["first_gap"].tolist())),
            third_stream_predicted_from_two=dict(zip(names, d["relation"].tolist())),
            pearson_between_streams=dict(zip(names, d["pearson"].tolist())),
            splitmix64_bits_changed_per_seed_bit=dict(
                zip(("mean", "min", "max"), d["avalanche"].tolist())),
            parallel=dict(workers=PAR_WORKERS, darts_per_worker=PAR_DARTS, trials=PAR_TRIALS,
                          jump_between_streams="2^40 steps",
                          rms_error_of_mean=four.tolist(), rms_error_one_worker=one.tolist(),
                          correlation_between_workers=d["worker_corr"].tolist()),
            limitations="The LCG's streams are related exactly (stream 3 = 2 x stream 2 "
                        "- stream 1 mod 2^32) but their Pearson correlation is near "
                        "zero: the usual coefficient does not see it. Hashing the seed "
                        "removes the relation between seeds 1, 2, 3; it does not make "
                        "an LCG's streams independent in general.",
            designs_tried=[
                "considered a lagged-Fibonacci generator filled naively from the seed to "
                "get streams with a large Pearson correlation; kept to the LCG already "
                "drawn in animation 110 and report its Pearson correlation as measured",
                "as built: fixed before measuring",
            ],
        ),
    )


# ====================================================================== 112: sampling
TOKENS = ("mat", "floor", "sofa", "roof", "moon")
TOKEN_P = (0.40, 0.25, 0.20, 0.10, 0.05)      # written out for the example, not measured
SAMP_N = 200_000
SAMP_REPEATS = 200
SAMP_GRID = tuple(int(10 * 2 ** k) for k in range(15))        # 10 .. 163,840 draws
SAMP_SHOWN = 3000                             # points kept for the drawings
BOUND = 16.0 / 9.0                            # the highest point of target()
BELL_EDGES = np.linspace(-4.0, 4.0, 41)


def inverse_cdf(u, p):
    """The outcome each ``u`` in [0, 1) falls on when the probabilities ``p`` are laid
    end to end along the unit interval."""
    edges = np.cumsum(p)
    return np.minimum(np.searchsorted(edges, u, side="right"), len(edges) - 1)


def target(x):
    """The density rejection sampling draws from: 12 x (1 - x)^2 on [0, 1]."""
    x = np.asarray(x, np.float64)
    return 12.0 * x * (1.0 - x) ** 2


def box_muller(u1, u2):
    """Two independent standard normal values from two uniform ones."""
    r = np.sqrt(-2.0 * np.log(1.0 - np.asarray(u1, np.float64)))
    angle = 2.0 * np.pi * np.asarray(u2, np.float64)
    return r * np.cos(angle), r * np.sin(angle)


def total_variation(counts, p):
    """Half the summed absolute difference between observed shares and ``p``."""
    c = np.asarray(counts, np.float64)
    return float(0.5 * np.abs(c / c.sum() - np.asarray(p).ravel()).sum())


def _token_errors(idx):
    """Error of the token shares, and of the shares of successive pairs of tokens,
    after each number of draws in SAMP_GRID."""
    p = np.asarray(TOKEN_P)
    pp = np.outer(p, p).ravel()
    pair = idx[:-1] * len(p) + idx[1:]
    single = [total_variation(np.bincount(idx[:n], minlength=len(p)), p) for n in SAMP_GRID]
    pairs = [total_variation(np.bincount(pair[:n - 1], minlength=len(pp)), pp)
             for n in SAMP_GRID]
    return single, pairs, np.bincount(pair, minlength=len(pp)).reshape(len(p), len(p))


def low_byte_unit(x):
    """The lowest byte of each output as a number in [0, 1): what ``x % 256`` keeps."""
    return (np.asarray(x, np.uint64) & np.uint64(255)).astype(np.float64) / 256.0


def _sampling_build():
    out = {}
    p = np.asarray(TOKEN_P)
    u = unit(pcg32_seeded(7, SAMP_N))
    idx = inverse_cdf(u, p)
    out["token_u"], out["token_idx"] = u[:24], idx[:24]
    out["token_counts"] = np.bincount(idx, minlength=len(p))
    # rejection: points under the curve are kept
    u = unit(pcg32_seeded(8, 2 * SAMP_N))
    x, y = u[0::2], u[1::2] * BOUND
    keep = y < target(x)
    out["rej_x"], out["rej_y"], out["rej_keep"] = x[:SAMP_SHOWN], y[:SAMP_SHOWN], keep[:SAMP_SHOWN]
    out["rej_rate"] = np.array(keep.mean())
    out["rej_hist"] = np.histogram(x[keep], 40, (0.0, 1.0), density=True)[0]
    # Box-Muller: pairs of uniforms become pairs of normal values
    u = unit(pcg32_seeded(9, 2 * SAMP_N))
    z1, z2 = box_muller(u[0::2], u[1::2])
    out["bm_u"] = np.stack([u[0:2 * SAMP_SHOWN:2], u[1:2 * SAMP_SHOWN:2]], 1)
    out["bm_z"] = np.stack([z1[:SAMP_SHOWN], z2[:SAMP_SHOWN]], 1)
    z = np.concatenate([z1, z2])
    out["bm_stats"] = np.array([z.mean(), z.std(), np.mean(np.abs(z) < 1.0),
                                np.corrcoef(z1, z2)[0, 1]])
    out["bm_hist"] = np.histogram(z, BELL_EDGES, density=True)[0]
    # how fast the shares of n draws approach the distribution
    nmax = SAMP_GRID[-1]
    single = np.zeros((SAMP_REPEATS, len(SAMP_GRID)))
    pairs = np.zeros_like(single)
    for r in range(SAMP_REPEATS):
        idx = inverse_cdf(np.random.default_rng(r).random(nmax), p)
        single[r], pairs[r], matrix = _token_errors(idx)
        if r == 0:
            out["pair_good"] = matrix
    out["err_single"] = np.stack([single.mean(0), np.percentile(single, 10, 0),
                                  np.percentile(single, 90, 0)])
    out["err_pairs"] = np.stack([pairs.mean(0), np.percentile(pairs, 10, 0),
                                 np.percentile(pairs, 90, 0)])
    out["slope"] = np.array(np.polyfit(np.log(SAMP_GRID), np.log(single.mean(0)), 1)[0])
    # the low byte of the 32-bit LCG used as the uniform number, and of PCG32 as a control
    for name, kind in (("lcg", "lcg32"), ("pcg", "pcg32")):
        rows = []
        for seed in (1, 2, 3):
            idx = inverse_cdf(low_byte_unit(stream(kind, seed, nmax)[1]), p)
            s, pr, matrix = _token_errors(idx)
            rows.append((s, pr))
            if seed == 1:
                out["pair_" + name] = matrix
                out["tokens_" + name] = idx[:32]
                out["period_" + name] = np.array(sequence_period(
                    idx[:8192], [2 ** k for k in range(12)]))
        out["err_single_" + name] = np.array([r[0] for r in rows])
        out["err_pairs_" + name] = np.array([r[1] for r in rows])
    return out


def sampling_data():
    """Everything ``sampling-a-distribution`` draws (statistics of generators' outputs)."""
    return cached_npz("random-sampling", _settings(n=SAMP_N, repeats=SAMP_REPEATS,
                                                   grid=SAMP_GRID, p=TOKEN_P),
                      _sampling_build)


def sampling():
    d = sampling_data()
    grid = np.asarray(SAMP_GRID, float)
    p = np.asarray(TOKEN_P)
    panels = (
        Panel("Tokens picked by the inverse-CDF method",
              "A written-out distribution over five tokens, and the shares of %d draws "
              "from it." % SAMP_N, "bars",
              dict(labels=np.array(TOKENS), values=d["token_counts"] / float(SAMP_N),
                   probability=p)),
        Panel("Rejection sampling",
              "Points thrown evenly into a box; those under the curve are kept "
              "(%.4f of them; exactly 9/16 expected)." % float(d["rej_rate"]), "points",
              dict(xy=np.stack([d["rej_x"], d["rej_y"]], 1), kept=d["rej_keep"].astype(float),
                   histogram=d["rej_hist"])),
        Panel("Box-Muller",
              "Pairs of even numbers turned into pairs of normal ones.", "points",
              dict(xy=d["bm_z"], uniform=d["bm_u"],
                   histogram=d["bm_hist"], stats=d["bm_stats"])),
        Panel("Error against the number of draws",
              "Total-variation distance between the shares of n draws and the "
              "distribution: mean of %d repeats; then the same for successive pairs, "
              "with numpy's generator and with the low byte of a 32-bit LCG."
              % SAMP_REPEATS, "curve",
              dict(x=grid, ys=np.stack([d["err_single"][0], d["err_pairs"][0],
                                        d["err_pairs_lcg"].mean(0)]),
                   labels=np.array(["single draws, numpy", "pairs, numpy",
                                    "pairs, LCG low byte"]),
                   single_band=d["err_single"][1:], pairs_band=d["err_pairs"][1:],
                   pairs_lcg=d["err_pairs_lcg"], pairs_pcg_low_byte=d["err_pairs_pcg"],
                   single_lcg=d["err_single_lcg"])),
    )
    return Study(
        "Sampling from a distribution",
        "From even numbers in [0, 1) to draws from any distribution",
        panels,
        metadata=dict(
            evidence=dict(written_out="the five tokens and their probabilities are an "
                          "example, not measured", statistics="everything else: outputs "
                          "of generators, deterministic given the seeds"),
            tokens=dict(zip(TOKENS, TOKEN_P)),
            seeds=dict(tokens=7, rejection=8, box_muller=9,
                       repeats="numpy.random.default_rng(0 .. %d)" % (SAMP_REPEATS - 1),
                       low_byte=[1, 2, 3]),
            draws=SAMP_N,
            rejection=dict(density="12 x (1 - x)^2", box_height=BOUND,
                           kept=float(d["rej_rate"]), expected=9.0 / 16.0),
            box_muller=dict(zip(("mean", "sd", "within_one_sd", "correlation"),
                                d["bm_stats"].tolist())),
            error_slope=float(d["slope"]),
            error=dict(n=list(SAMP_GRID), mean=d["err_single"][0].tolist()),
            pair_error=dict(numpy=d["err_pairs"][0].tolist(),
                            lcg_low_byte=d["err_pairs_lcg"].tolist(),
                            pcg_low_byte=d["err_pairs_pcg"].tolist()),
            lcg_low_byte_token_period=int(d["period_lcg"]),
            limitations="The low byte of the LCG visits every byte once in 256 draws, so "
                        "its single-token shares are closer to the distribution than "
                        "chance would make them; the fault shows only in which token "
                        "follows which. PCG32's low byte is the control: the fault is "
                        "the generator's, not the use of a low byte (a byte gives 256 "
                        "levels, which rounds the probabilities and leaves a small "
                        "floor of its own, seen in the control).",
            designs_tried=[
                "considered a 16-bit LCG and RANDU as the bad generator: their "
                "one-dimensional histograms are fine, so nothing would show",
                "as built: the low byte of the 32-bit LCG (the x % 256 mistake), judged "
                "on successive pairs; fixed before the curves were computed",
            ],
        ),
    )


# ====================================================================== local measurements
LOCAL_NOTE = ("measured once on the machine that built the cache; these numbers are that "
              "machine's and differ on every other")


def _local_build():
    n = N_BYTES
    stamps = np.empty(n + 1, np.int64)
    wall = np.empty(n, np.int64)
    tick, clock = time.perf_counter_ns, time.time_ns
    for i in range(n):
        stamps[i] = tick()
        wall[i] = clock()
    stamps[n] = tick()
    return dict(
        stamps=stamps[:9], deltas=np.diff(stamps), clock_high=((wall >> 32) & 255),
        os_bytes=np.frombuffer(os.urandom(n), np.uint8).copy(),
        machine=np.array("%s %s" % (platform.system(), platform.machine())),
        measured=np.array(time.strftime("%Y-%m-%d")))


def local_data():
    """This machine's raw measurements: ``deltas`` (ns between successive readings of the
    high-resolution timer, with one reading of the wall clock in between), ``clock_high``
    (the wall clock's nanoseconds shifted right 32 bits, low byte), ``os_bytes``
    (``os.urandom``). Not reproducible: see LOCAL_NOTE."""
    return cached_npz("random-local", _settings(n=N_BYTES), _local_build)


def byte_measures(b):
    """The single-source estimates, each in bits per byte: single-byte Shannon entropy,
    min-entropy, Shannon entropy given the previous byte, given the previous two, and
    the size after zlib."""
    return np.array([byte_entropy(b), min_entropy(b), conditional_entropy(b, 1),
                     conditional_entropy(b, 2), zlib_bits(b)])


def local_sources():
    """The four local byte sources of ``entropy-sources``: the low byte of the timer's
    gaps, the operating system's bytes, a counter, the clock's high bits."""
    d = local_data()
    n = len(d["deltas"])
    return (
        (d["deltas"] & 255).astype(np.uint8), d["os_bytes"],
        (np.arange(n) & 255).astype(np.uint8), d["clock_high"].astype(np.uint8),
    )


# ====================================================================== 113: sources
SRC_NAMES = ("timer jitter, low byte", "operating system", "a counter", "clock, high bits")
BIAS = 0.8                                    # the biased coin that is cleaned and mixed
VN_SEEDS = (0, 1, 2)
XOR_STREAMS = 6
STRETCH_BITS = (4, 8, 12, 16, 20)


def biased_bits(seed, n=N_BYTES):
    """``n`` independent bits that are 1 with probability BIAS (numpy's generator)."""
    return (np.random.default_rng(seed).random(n) < BIAS).astype(np.uint8)


def cleaning(bits_in):
    """Von Neumann's trick on a bit stream: share of ones and correlation of successive
    bits, before and after, and output bits per input bit."""
    b = np.asarray(bits_in, np.uint8)
    out = von_neumann(b)

    def lag(v):
        v = v.astype(np.float64)
        return float(np.corrcoef(v[:-1], v[1:])[0, 1]) if v.std() > 0 else 0.0
    return np.array([b.mean(), out.mean() if len(out) else 0.0, len(out) / float(len(b)),
                     lag(b), lag(out) if len(out) > 2 else 0.0])


def _sources_build():
    out = {}
    out["vn"] = np.array([cleaning(biased_bits(s)) for s in VN_SEEDS])
    demo = biased_bits(VN_SEEDS[0])[:32]
    out["vn_in"], out["vn_out"] = demo, von_neumann(demo)
    # XOR of 1 .. 6 independent biased streams
    acc = np.zeros(N_BYTES, np.uint8)
    share = []
    for k in range(XOR_STREAMS):
        acc ^= biased_bits(10 + k)
        share.append(acc.mean())
    out["xor"] = np.array(share)
    out["xor_exact"] = np.array([(1 - (1 - 2 * BIAS) ** (k + 1)) / 2
                                 for k in range(XOR_STREAMS)])
    same = biased_bits(10) ^ biased_bits(10)
    out["xor_same"] = np.array(same.mean())
    # a hash: one input bit changed, how many of the 256 output bits change
    rng = np.random.default_rng(0)
    changed = []
    for _ in range(1000):
        msg = bytearray(rng.integers(0, 256, 64, dtype=np.uint8).tobytes())
        a = hashlib.sha256(msg).digest()
        at = int(rng.integers(0, 512))
        msg[at // 8] ^= 1 << (at % 8)
        b = hashlib.sha256(msg).digest()
        changed.append(sum(bin(p ^ q).count("1") for p, q in zip(a, b)))
    out["hash_changed"] = np.array([np.mean(changed), np.min(changed), np.max(changed)])
    # stretching: every seed of k bits, the first 64 bits of its stream
    distinct = []
    for k in STRETCH_BITS:
        state = splitmix64_array(np.arange(1 << k, dtype=np.uint64))
        w0 = pcg32_parts(state)[2]
        w1 = pcg32_parts(state * np.uint64(PCG_MULT) + np.uint64(PCG_INC))[2]
        distinct.append(len(np.unique((w0 << np.uint64(32)) | w1)))
    out["distinct"] = np.array(distinct, np.int64)
    stretched = pcg32_seeded(12345, N_BYTES // 4).astype("<u4").view(np.uint8)
    out["stretched"] = byte_measures(stretched)
    out["stretched_bytes"] = stretched[:4096].copy()
    return out


def sources_data():
    """The reproducible part of ``entropy-sources`` (seeded streams, exact counts)."""
    return cached_npz("random-sources", _settings(n=N_BYTES, bias=BIAS, seeds=VN_SEEDS,
                                                  streams=XOR_STREAMS, bits=STRETCH_BITS),
                      _sources_build)


def pool_seed(local=None):
    """The 256-bit seed made by hashing a pool of this machine's sources (local)."""
    jitter, os_bytes, counter, clock = local_sources() if local is None else local
    pool = b"".join(np.asarray(s[:32], np.uint8).tobytes()
                    for s in (jitter, os_bytes, counter, clock))
    return pool, hashlib.sha256(pool).digest()


def entropy_sources():
    d = sources_data()
    loc = local_data()
    srcs = local_sources()
    measures = np.array([byte_measures(s)[:3] for s in srcs])
    ideal = byte_measures(np.random.default_rng(0).integers(0, 256, N_BYTES, dtype=np.uint8))
    jitter_bit = (loc["deltas"] & 1).astype(np.uint8)
    vn_local = cleaning(jitter_bit)
    vn = d["vn"]
    entropy = np.array([[coin_entropy(r[0]), coin_entropy(r[1])] for r in vn])
    panels = (
        Panel("This machine's sources (local)",
              "Rows: timer jitter (low byte of the gap between readings), the operating "
              "system's bytes, a counter, the clock's high bits. Columns, in bits per "
              "byte, estimated from %d bytes: single-byte Shannon entropy, min-entropy, "
              "Shannon entropy given the previous byte." % N_BYTES, "matrix",
              dict(values=measures, ideal=ideal[:3], gaps_ns=loc["deltas"][:4096])),
        Panel("Von Neumann's trick on a biased coin",
              "Share of ones before and after, and bits kept per bit collected: a coin "
              "that shows 1 %.0f times in 100 (three seeds), then this machine's timer "
              "bit (local)." % (100 * BIAS), "bars",
              dict(labels=np.array(["ones before", "ones after", "kept per bit"]),
                   values=vn[:, :3].mean(0), all_seeds=vn, coin_entropy=entropy,
                   local_timer_bit=vn_local)),
        Panel("Mixing biased coins by XOR",
              "Share of ones in the XOR of 1 to %d independent biased streams, measured "
              "and exact." % XOR_STREAMS, "curve",
              dict(x=np.arange(1, XOR_STREAMS + 1).astype(float),
                   ys=np.stack([d["xor"], d["xor_exact"]]),
                   labels=np.array(["measured", "exact"]), same_stream=d["xor_same"],
                   hash_bits_changed=d["hash_changed"])),
        Panel("A generator only stretches its seed",
              "Different 64-bit beginnings over every seed of k bits: exactly 2^k.", "bars",
              dict(labels=np.array(["%d bits" % k for k in STRETCH_BITS]),
                   values=d["distinct"].astype(float), seed_bits=np.array(STRETCH_BITS),
                   stretched_estimates=d["stretched"])),
    )
    return Study(
        "Where the unpredictability comes from",
        "A generator stretches its seed; the seed has to come from somewhere",
        panels,
        metadata=dict(
            evidence=dict(
                local="panel 1 and the timer bit of panel 2: " + LOCAL_NOTE,
                statistics="the biased coins (numpy's generator, seeds %s and 10-15) and "
                           "the hash test" % (VN_SEEDS,),
                exact="the XOR formula; the count of streams over every seed"),
            local=True, machine=str(loc["machine"]), measured_on=str(loc["measured"]),
            samples=N_BYTES,
            sources=dict(zip(SRC_NAMES, (
                "time.perf_counter_ns() read in a loop with one time.time_ns() between "
                "readings; low byte of each gap in ns", "os.urandom", "0, 1, 2, ... mod 256",
                "time.time_ns() >> 32, low byte"))),
            estimates_bits_per_byte={n: dict(zip(("shannon_single", "min_entropy",
                                                  "shannon_given_previous"), m.tolist()))
                                     for n, m in zip(SRC_NAMES, measures)},
            ideal_source_at_this_sample_size=ideal[:3].tolist(),
            von_neumann=dict(bias=BIAS, seeds=list(VN_SEEDS),
                             ones_before=vn[:, 0].tolist(), ones_after=vn[:, 1].tolist(),
                             kept_per_bit=vn[:, 2].tolist(),
                             expected_kept=BIAS * (1 - BIAS),
                             local_timer_bit=dict(zip(
                                 ("ones_before", "ones_after", "kept_per_bit",
                                  "serial_correlation_before", "serial_correlation_after"),
                                 vn_local.tolist()))),
            xor=dict(measured=d["xor"].tolist(), exact=d["xor_exact"].tolist(),
                     a_stream_with_itself=float(d["xor_same"])),
            sha256_bits_changed_by_one_input_bit=dict(
                zip(("mean", "min", "max"), d["hash_changed"].tolist())),
            stretch=dict(seed_bits=list(STRETCH_BITS), distinct_streams=d["distinct"].tolist(),
                         generator="PCG32, state = splitmix64(seed)"),
            limitations="The estimates say how the bytes look, not how hard they are to "
                        "guess: a counter scores 8 bits per byte on single bytes. The "
                        "operating system's bytes come from its own generator, seeded "
                        "from hardware events; their estimates cannot tell them from "
                        "any good generator's. Von Neumann's trick assumes independent "
                        "bits, which timer bits need not be: their serial correlation "
                        "is reported. Timer jitter depends on the clock's resolution "
                        "and the machine's load.",
            designs_tried=["as built: fixed before measuring; the local numbers are "
                           "whatever this machine gave"],
        ),
    )


# ====================================================================== 114: measuring
ENT_NAMES = ("English text", "the text, compressed", "a generator", "a bad generator's low byte",
             "operating system", "a counter")
ENT_MEASURES = ("single bytes", "min-entropy", "given 1 before", "given 2 before", "zlib")
ENT_TESTS = ("chi-square", "serial correlation", "runs")
ENT_OS = 4                                    # the local row
SECRET = 3141592653                           # the generator's 32-bit seed
BIAS_GRID = tuple(1 << k for k in range(4, 19))
BIAS_REPEATS = 100
SHOWN = 3                                     # outputs used to recover the seed


def byte_tests(b):
    """Three tests as z-scores (:func:`chi_square_z`, :func:`serial_z`, :func:`runs_z`)."""
    return np.array([chi_square_z(b), serial_z(b), runs_z(b)])


def pcg16_first_matches(start, stop, target):
    """Every seed in ``[start, stop)`` (inside one 2^28 block) whose first PCG 32/16
    output is ``target``. All seeds of a block rotate by the same amount (their top four
    bits), so the target is rotated back once and compared with the unrotated bits."""
    r = start >> 28
    want = np.uint16(((target << r) | (target >> (16 - r))) & 0xFFFF)
    found = []
    chunk = 1 << 22
    low = np.arange(chunk, dtype=np.uint32)
    s, x = np.empty(chunk, np.uint32), np.empty(chunk, np.uint32)
    for at in range(start, stop, chunk):
        k = min(chunk, stop - at)
        np.add(low, np.uint32(at), out=s)
        np.right_shift(s, np.uint32(10), out=x)
        np.bitwise_xor(x, s, out=x)
        np.right_shift(x, np.uint32(12), out=x)
        found.append(s[:k][x[:k].astype(np.uint16) == want])
    return np.concatenate(found)


def recover_seed(outputs):
    """Try every 32-bit seed against the outputs in turn: ``(seeds still possible after
    0, 1, 2, ... outputs, the seeds left)``."""
    left = np.concatenate([pcg16_first_matches(r << 28, (r + 1) << 28, int(outputs[0]))
                           for r in range(16)])
    counts = [1 << 32, len(left)]
    state = left.copy()
    for want in outputs[1:]:
        state = state * np.uint32(PCG16_MULT) + np.uint32(PCG16_INC)
        keep = pcg16_output(state) == np.uint32(want)
        left, state = left[keep], state[keep]
        counts.append(len(left))
    return counts, left


def entropy_sources_fixed():
    """The five reproducible byte sources (the operating system's is local): None in
    its place."""
    from . import realdata

    raw = realdata.text("train").encode("utf-8")
    packed = zlib.compress(raw, 9)
    assert len(raw) >= N_BYTES and len(packed) >= N_BYTES + 2
    return [
        np.frombuffer(raw[:N_BYTES], np.uint8),
        np.frombuffer(packed[2:N_BYTES + 2], np.uint8),
        pcg16_bytes(SECRET, N_BYTES),
        (lcg_states(1, N_BYTES, *LCG32) & np.uint64(255)).astype(np.uint8),
        None,
        (np.arange(N_BYTES) & 255).astype(np.uint8),
    ]


def _bias_curve(draw):
    est = np.zeros((BIAS_REPEATS, len(BIAS_GRID)))
    for r in range(BIAS_REPEATS):
        sample = draw(np.random.default_rng(r), BIAS_GRID[-1])
        est[r] = [byte_entropy(sample[:n]) for n in BIAS_GRID]
    return np.stack([est.mean(0), est.min(0), est.max(0)])


def _entropy_build():
    out = {}
    srcs = entropy_sources_fixed()
    keep = [s for s in srcs if s is not None]
    out["measures"] = np.array([byte_measures(s) for s in keep])
    out["tests"] = np.array([byte_tests(s) for s in keep])
    out["strips"] = np.array([s[:4096] for s in keep])
    out["hist"] = np.array([block_counts(s, 1) for s in keep])
    out["pairs"] = np.array([block_counts(s, 2).reshape(256, 256) for s in keep]
                            ).astype(np.uint32)
    ideal = [np.random.default_rng(s).integers(0, 256, N_BYTES, dtype=np.uint8)
             for s in (0, 1, 2)]
    out["ideal"] = np.array([byte_measures(s) for s in ideal])
    out["ideal_tests"] = np.array([byte_tests(s) for s in ideal])
    out["generator_tests"] = np.array([byte_tests(pcg16_bytes(s, N_BYTES))
                                       for s in (SECRET, 1, 2)])
    out["generator_measures"] = np.array([byte_measures(pcg16_bytes(s, N_BYTES))
                                          for s in (SECRET, 1, 2)])
    # the single-byte estimate against the sample size, where the true value is known
    counts = out["hist"][0].astype(np.float64)
    p = counts / counts.sum()
    out["bias_uniform"] = _bias_curve(lambda g, n: g.integers(0, 256, n, dtype=np.uint8))
    out["bias_text"] = _bias_curve(lambda g, n: inverse_cdf(g.random(n), p))
    out["bias_true"] = np.array([8.0, shannon(counts)])
    # every seed tried against the generator's first outputs
    shown = pcg16(SECRET, SHOWN)
    began = time.perf_counter()
    counts, left = recover_seed(shown)
    out["search_seconds"] = np.array(time.perf_counter() - began)
    out["shown"] = shown.astype(np.int64)
    out["left"] = np.array(counts, np.int64)
    out["recovered"] = left.astype(np.int64)
    return out


def entropy_data():
    """Everything ``measuring-entropy`` draws, the operating system's row filled in from
    :func:`local_data` (local: see LOCAL_NOTE)."""
    d = dict(cached_npz("random-entropy", _settings(n=N_BYTES, secret=SECRET, grid=BIAS_GRID,
                                                    repeats=BIAS_REPEATS, shown=SHOWN),
                        _entropy_build))
    os_bytes = local_data()["os_bytes"]

    def insert(key, value):
        d[key] = np.insert(d[key], ENT_OS, np.asarray(value, d[key].dtype), axis=0)
    insert("measures", byte_measures(os_bytes))
    insert("tests", byte_tests(os_bytes))
    insert("strips", os_bytes[:4096])
    insert("hist", block_counts(os_bytes, 1))
    insert("pairs", block_counts(os_bytes, 2).reshape(256, 256))
    return d


def measuring_entropy():
    from . import realdata

    d = entropy_data()
    loc = local_data()
    m = d["measures"]
    total_bits = 32.0
    panels = (
        Panel("Five estimates for six byte sources",
              "Rows: English text, the text compressed, a 32-bit generator, a bad "
              "generator's low byte, the operating system (local), a counter. Columns, "
              "bits per byte from %d bytes: single-byte Shannon entropy, min-entropy, "
              "Shannon entropy given one and given two bytes before, size after zlib."
              % N_BYTES, "matrix",
              dict(values=m, ideal=d["ideal"].mean(0), histograms=d["hist"])),
        Panel("The single-byte estimate is low for small samples",
              "Mean of %d samples of each size: bytes with equal shares (true value 8) "
              "and bytes drawn independently with the text's shares (true value %.3f)."
              % (BIAS_REPEATS, float(d["bias_true"][1])), "curve",
              dict(x=np.asarray(BIAS_GRID, float),
                   ys=np.stack([d["bias_uniform"][0], d["bias_text"][0]]),
                   labels=np.array(["equal shares", "the text's shares"]),
                   true=d["bias_true"], uniform_range=d["bias_uniform"][1:],
                   text_range=d["bias_text"][1:])),
        Panel("Three statistical tests",
              "z-scores: chi-square of the byte counts, serial correlation, runs above "
              "and below the median. Beyond 3 either way fails.", "matrix",
              dict(values=d["tests"], ideal=d["ideal_tests"],
                   generator_seeds=d["generator_tests"])),
        Panel("Recovering the generator's seed",
              "Seeds still possible after 0, 1, 2 and 3 of its 16-bit outputs, every "
              "one of 2^32 tried.", "bars",
              dict(labels=np.array(["none", "1 output", "2 outputs", "3 outputs"]),
                   values=d["left"].astype(float), outputs=d["shown"],
                   recovered=d["recovered"])),
    )
    return Study(
        "Measuring entropy",
        "What an estimate from the output can and cannot see",
        panels,
        metadata=dict(
            evidence=dict(
                real_text="rows 1 and 2: the first %d bytes of the training split of "
                          "Tiny Shakespeare, and of its zlib stream" % N_BYTES,
                statistics="the generators' rows, the sample-size curves",
                local="the operating system's row: " + LOCAL_NOTE,
                exact="the seed search: every 32-bit seed tried"),
            dataset=realdata.text_metadata("train"),
            local=True, machine=str(loc["machine"]), measured_on=str(loc["measured"]),
            samples=N_BYTES, sources=list(ENT_NAMES), measures=list(ENT_MEASURES),
            which_entropy=dict(zip(ENT_MEASURES, (
                "plug-in estimate of the Shannon entropy of one byte",
                "minus log2 of the commonest byte's share",
                "plug-in estimate of the Shannon entropy of a byte given the one before",
                "the same given the two before",
                "bits per byte after zlib level 9: an upper bound on the entropy rate"))),
            estimates={n: row.tolist() for n, row in zip(ENT_NAMES, m)},
            ideal_source_at_this_sample_size=d["ideal"].mean(0).tolist(),
            tests={n: row.tolist() for n, row in zip(ENT_NAMES, d["tests"])},
            generator=dict(kind="PCG XSH-RR 32/16, the state started at the seed",
                           seed=SECRET, seeds_tested=[SECRET, 1, 2],
                           tests_by_seed=d["generator_tests"].tolist(),
                           estimates_by_seed=d["generator_measures"].tolist(),
                           true_entropy_bits=total_bits,
                           true_bits_per_byte=total_bits / N_BYTES),
            bad_generator="x -> 1664525 x + 1013904223 mod 2^32 from 1, lowest byte",
            small_samples=dict(sizes=list(BIAS_GRID), repeats=BIAS_REPEATS,
                               true=d["bias_true"].tolist(),
                               uniform=d["bias_uniform"][0].tolist(),
                               text_shares=d["bias_text"][0].tolist()),
            seed_search=dict(outputs=d["shown"].tolist(), seeds_left=d["left"].tolist(),
                             recovered=d["recovered"].tolist(),
                             seconds_on_the_machine_that_built_it=float(d["search_seconds"])),
            limitations="Plug-in estimates are biased low, the more so the more symbols "
                        "there are: with %d bytes there are 65,536 possible pairs and "
                        "16.7 million triples, so even an ideal source scores below 8 "
                        "given one byte before and far below given two (the ideal row "
                        "says how far). The three tests are the simplest of their "
                        "kind. No estimate from the output bounds a generator's "
                        "entropy from above except by finding the seed." % N_BYTES,
            designs_tried=["as built: sources, estimators, tests and the generator's "
                           "seed fixed before measuring"],
        ),
    )


STUDIES = {
    "pseudo-random-generators": generators,
    "seeds-and-streams": seeds,
    "sampling-a-distribution": sampling,
    "entropy-sources": entropy_sources,
    "measuring-entropy": measuring_entropy,
}

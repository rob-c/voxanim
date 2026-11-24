"""Unfolding: running a detector's smearing backwards. The program of the unfolding animation
of "Blocks and code".

    python examples/unfolding.py          # needs numpy

Made-up events throughout, with a fixed seed. The true spectrum: 10,000 events of energy 0 to
100 GeV in 10 bins of 10 GeV, three quarters falling (exp(-E / 25 GeV)) and a quarter a bump
at 60 GeV (a Gaussian 10 GeV wide), both cut to 0-100 GeV.

The made-up detector finds an event with a chance rising from 60% at 0 GeV to 90% at 100
GeV, and measures its energy off by a Gaussian 10 GeV wide, as wide as a bin; an event
measured below 0 or above 100 GeV is lost too.

1. The response matrix R, built from simulation as everything in HEP is: 1,000,000 simulated
   events (true energies falling as exp(-E / 30 GeV), kept below 100 GeV), each with its true
   and its measured energy. R[i, j] is the fraction of the events simulated in true bin j
   that are found and measured in bin i.
2. What the detector measures: R @ truth expected, and Poisson counts round it (the data).
3. Inverting R: the expected counts give the truth back; the data's small noise comes back
   huge, since R's small singular values divide it.
4. Iterative Bayesian unfolding (D'Agostini): from the simulation's spectrum, each iteration
   folds the guess with R, compares with the data and corrects. Few iterations: close to the
   simulation (bias); many: close to the inversion (spread). 2,000 pseudo-experiments
   measure both.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: uf-imports
import math

import numpy as np

# endregion

rng = np.random.default_rng(1)
N = 10_000                                # true events in the made-up spectrum
SIMULATED = 1_000_000                     # simulated events, before the 100 GeV cut
TOYS = 2_000                              # pseudo-experiments


def falling(edges, slope=25.0):
    """The fraction of exp(-E / slope), cut to the bins, in each bin."""
    c = 1.0 - np.exp(-edges / slope)
    return np.diff(c) / (c[-1] - c[0])


def bump(edges, at=60.0, width=10.0):
    """The fraction of a Gaussian bump, cut to the bins, in each bin."""
    c = np.array([math.erf((e - at) / (width * math.sqrt(2.0))) for e in edges])
    return np.diff(c) / (c[-1] - c[0])


# region: uf-truth
edges = np.linspace(0, 100, 11)           # 10 bins of 10 GeV
truth = N * (0.75 * falling(edges)        # a falling spectrum
             + 0.25 * bump(edges))        # with a bump at 60 GeV
# endregion

# region: uf-simulate
t = rng.exponential(30, SIMULATED)        # true energies, simulated
t = t[t < 100]                            # in our 10 bins
p = 0.6 + 0.003 * t                       # its chance to be found
found = rng.random(t.size) < p            # found by the detector?
r = t + rng.normal(0, 10, t.size)         # measured: smeared 10 GeV
# endregion

# region: uf-response
H = np.histogram2d(r[found], t[found],
                   [edges, edges])[0]     # measured bin x true bin
n = np.histogram(t, edges)[0]             # simulated, per true bin
R = H / n                                 # column j: where j lands
# endregion

# region: uf-measure
expected = R @ truth                      # what the detector should see
data = rng.poisson(expected)              # what it sees: counts
# endregion

# region: uf-invert
back = np.linalg.solve(R, expected)       # no noise: the truth again
naive = np.linalg.solve(R, data)          # with noise: wild
# endregion

# region: uf-svd
U, s, Vt = np.linalg.svd(R)               # R = U @ diag(s) @ Vt
noise = U.T @ (data - expected)           # the noise along each U[:, k]
blown = noise / s                         # comes back divided by s
# endregion

# region: uf-prior
eff = R.sum(0)                            # found and kept, per true bin
prior = n * data.sum() / (R @ n).sum()    # the simulation, scaled
# endregion


# region: uf-bayes
def bayes(data, prior, iterations):
    t = prior * np.ones(data.shape)       # start: the simulation's shape
    for _ in range(iterations):
        fold = t @ R.T                    # what t would be measured as
        t = t * ((data / fold) @ R) / eff
    return t
# endregion


def pseudo_experiments(iterations=(0, 1, 2, 4, 10, 20, 50)):
    """Bias and spread of the unfolded spectrum, root mean square over the 10 bins, for each
    number of iterations, over TOYS pseudo-experiments (and the inversion's)."""
    out = {}
    # region: uf-toys
    toys = rng.poisson(expected, (TOYS, 10))  # pseudo-experiments
    for k in iterations:
        u = bayes(toys, prior, k)         # all 2,000 at once
        bias = u.mean(0) - truth          # off, on average
        spread = u.std(0)                 # scatter, toy to toy
        out[k] = bias, spread
    # endregion
    inv = np.linalg.solve(R, toys.T).T
    out["inverse"] = inv.mean(0) - truth, inv.std(0)
    return out


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def its(k):
    return "iteration" if k == 1 else "iterations"


def printed():
    """What the program prints, line by line (short lines, as the animation's output slot
    shows them). Call it once: the pseudo-experiments draw from ``rng``."""
    out = ["truth total %d; 50-60 GeV: %d" % (round(truth.sum()), round(truth[5])),
           "simulated %d, counted %d" % (SIMULATED, H.sum()),
           "expected %d, data %d" % (round(expected.sum()), data.sum()),
           "back - truth: %.1e" % np.abs(back - truth).max(),
           "naive: %d to %d" % (round(naive.min()), round(naive.max())),
           "s: %.3f down to %.4f" % (s[0], s[-1])]
    for k in (0, 1, 4, 10):
        out.append("%d %s: 50-60 GeV holds %d" % (k, its(k), round(bayes(data, prior, k)[5])))
    for k, (bias, spread) in pseudo_experiments().items():
        out.append("%s: bias %.0f, spread %.0f" % (
            "R inverted" if k == "inverse" else "%d %s" % (k, its(k)), rms(bias), rms(spread)))
    return out


if __name__ == "__main__":
    for line in printed():
        print(line)

"""Error bars on a network: the fit of examples/minuit_vs_sgd.py trained by SGD 100 times with
new seeds, and 100 times on resampled data, against HESSE's errors. The program of the
animation "Error bars on a network" of "Blocks and code".

    python examples/ensembles.py          # needs numpy and iminuit

Made-up data, the same 40 points as examples/minuit_vs_sgd.py: y = a exp(-x / tau) + b with
a = 5, tau = 2.5, b = 1, plus Gaussian noise of 0.3 (seed 1).

1. One SGD run: 3,000 steps from (1, 1, 0), each on 8 of the 40 points, learning rate 0.005.
   Its answer is where it ends.
2. 100 runs with new seeds, all on the same 40 points: the answers differ only by the luck of
   the batches (the optimiser's noise).
3. 100 runs on bootstrap resamples: each run's data are 40 picks from the 40 points, repeats
   allowed, so some points count twice and some not at all. The answers' spread is an
   ensemble's error bar.
4. Minuit's HESSE on the 40 points: its errors, to compare.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: en-imports
import numpy as np
from iminuit import Minuit

# endregion

# region: en-data
rng = np.random.default_rng(1)
x = np.linspace(0, 10, 40)                # 40 places
sigma = 0.3                               # the noise on each point
y = 5.0 * np.exp(-x / 2.5) + 1.0          # the truth: a=5, tau=2.5, b=1
y = y + rng.normal(0, sigma, x.size)      # and measured, with noise
# endregion


def gradient(p, idx):
    """The mean of the chi-square terms of points ``idx``, and its gradient (as in
    minuit_vs_sgd.py)."""
    a, tau, b = p
    e = np.exp(-x[idx] / tau)
    r = (y[idx] - (a * e + b)) / sigma
    g = np.array([(-2 * r * e).mean(), (-2 * r * a * e * x[idx] / tau ** 2).mean(),
                  (-2 * r).mean()]) / sigma
    return g, float((r ** 2).mean())


# region: en-train
def train(points, seed):                  # one SGD run
    pick = np.random.default_rng(seed)
    p = np.array([1.0, 1.0, 0.0])         # the same start
    for _ in range(3000):
        k = pick.permutation(40)[:8]      # 8 of its 40 points
        g, _ = gradient(p, points[k])
        p = p - 0.005 * g                 # a small step downhill
    return p                              # its answer: where it ends
# endregion


def ensembles():
    # region: en-seeds
    everyone = np.arange(40)              # all 40 points, every run
    seeds = []
    for seed in range(100):               # a new seed each run
        seeds.append(train(everyone, seed))
    print(np.std(seeds, 0, ddof=1).round(3))
    # endregion
    # region: en-bootstrap
    draw = np.random.default_rng(2)
    boots = []
    for seed in range(100, 200):          # a new resample each run
        points = draw.integers(0, 40, 40)  # 40 picks, repeats allowed
        boots.append(train(points, seed))
    # endregion
    # region: en-spread
    boots = np.array(boots)
    print(boots.std(0, ddof=1).round(2))  # the spread: error bars
    print(np.cov(boots.T).round(3))       # and how they move together
    # endregion
    return np.array(seeds), boots


# region: en-chi2
def chi2(a, tau, b):                      # all 40 points
    fit = a * np.exp(-x / tau) + b
    r = (y - fit) / sigma
    return float((r ** 2).sum())
# endregion


def run_minuit():
    # region: en-hesse
    m = Minuit(chi2, a=1.0, tau=1.0, b=0.0)
    m.errordef = Minuit.LEAST_SQUARES     # chi-square: error at +1
    m.migrad()                            # find the minimum
    m.hesse()                             # curvature -> error bars
    print(np.round(m.errors, 2))
    # endregion
    return m


if __name__ == "__main__":
    seeds, boots = ensembles()
    m = run_minuit()
    print("seeds:     mean", seeds.mean(0).round(3), "spread", seeds.std(0, ddof=1).round(3))
    print("resamples: mean", boots.mean(0).round(3), "spread", boots.std(0, ddof=1).round(3))
    print("HESSE:     values", np.round(m.values, 3), "errors", np.round(m.errors, 3))
    print("HESSE covariance:\n", np.round(np.array(m.covariance), 4))

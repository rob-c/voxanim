"""The Asimov dataset: one made-up dataset in which every bin holds exactly its expected count
gives the expected (median) significance, without thousands of toy experiments. The program of
the animation "The Asimov dataset" of "Blocks and code".

    python examples/asimov.py             # needs numpy

A made-up binned analysis: 4 bins of a mass, with an expected background b and an expected
signal s in each (made-up numbers, no detector). A real experiment sees counts n drawn from
Poisson(s + b) in each bin.

1. One experiment's discovery significance: fit the signal strength mu (0 = background only,
   1 = the signal expected; never below 0) by maximising the Poisson likelihood, then
   q0 = 2 sum(n ln(1 + mu s / b) - mu s) and Z = sqrt(q0).
2. 10,000 toy experiments (pseudo-experiments): counts drawn at random, each one's Z, and
   their median: the expected significance.
3. The Asimov dataset: n = s + b exactly, in every bin. Its Z is
   sqrt(2 sum((s + b) ln(1 + s / b) - s)), and it lands on the toys' median.
4. The simple s / sqrt(b), bin by bin: close when b is large, far too high when b is small.
   Done again with a tenth of the background.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: as-imports
import numpy as np

# endregion

# region: as-counts
s = np.array([3.0, 8.0, 8.0, 3.0])        # signal expected, per bin
b = np.array([30.0, 20.0, 12.0, 8.0])     # background expected
# endregion


# region: as-fit
def fit_mu(n, s, b):                      # the best signal strength
    mu = 1.0
    for _ in range(20):                   # Newton's method
        m = mu * s + b                    # expected counts at mu
        slope = (n * s / m - s).sum()     # of the log-likelihood
        bend = (n * s * s / m ** 2).sum()  # and its curvature
        mu = max(mu + slope / bend, 0.0)  # never below 0
    return mu
# endregion


# region: as-z
def z0(n, s, b):                          # one experiment's significance
    extra = fit_mu(n, s, b) * s           # the signal fitted, per bin
    t = n * np.log(1 + extra / b) - extra
    return np.sqrt(2 * t.sum())           # sqrt(q0)
# endregion


def expected(s, b, seed):
    """The median of 10,000 toys' Z, the Asimov dataset's Z and s / sqrt(b)."""
    # region: as-toys
    rng = np.random.default_rng(seed)
    toys = rng.poisson(s + b, (10000, 4))  # 10,000 made-up experiments
    z = [z0(n, s, b) for n in toys]
    print(np.median(z).round(2))          # the median toy
    # endregion
    # region: as-asimov
    asimov = s + b                        # every bin exactly as expected
    print(z0(asimov, s, b).round(2))      # one dataset, no toys
    # endregion
    # region: as-simple
    simple = np.sqrt((s ** 2 / b).sum())  # s / sqrt(b), bin by bin
    print(simple.round(2))
    # endregion
    return toys, np.array(z), z0(asimov, s, b), simple


if __name__ == "__main__":
    # region: as-less
    for back, seed in ((b, 7), (b / 10, 8)):
        print("background", back)         # then a tenth of it
        expected(s, back, seed)
    # endregion

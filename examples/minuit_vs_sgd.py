"""How Minuit feels its way to a minimum, and how SGD stumbles there: the program of the Minuit
animations of "Blocks and code".

    python examples/minuit_vs_sgd.py          # needs numpy and iminuit

Made-up data: 40 points on a falling exponential with a flat background, y = a exp(-x / tau)
+ b with a = 5, tau = 2.5, b = 1, plus Gaussian noise of 0.3. The model has three parameters,
so the loss (chi-square) lives over a 3-D space of (a, tau, b).

1. Minuit's MIGRAD, started at (1, 1, 0). Every point it asks the model about is recorded: around
   each point it steps a little up and down along each of the 3 axes to feel the slope, then
   steps, checks, and repeats; HESSE then measures the curvature for the error bars.
2. SGD from the same start: each step looks at 8 of the 40 points chosen at random, so its
   direction is noisy; it takes many small steps, and the noise averages out.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: mn-imports
import numpy as np
from iminuit import Minuit

# endregion

# region: mn-data
rng = np.random.default_rng(1)
x = np.linspace(0, 10, 40)                # 40 places
sigma = 0.3                               # the noise on each point
y = 5.0 * np.exp(-x / 2.5) + 1.0          # the truth: a=5, tau=2.5, b=1
y = y + rng.normal(0, sigma, x.size)      # and measured, with noise
# endregion

asked = []                                # every point Minuit asks about


# region: mn-model
def model(x, a, tau, b):                  # 3 numbers to find
    return a * np.exp(-x / tau) + b
# endregion


# region: mn-chi2
def chi2(a, tau, b):                      # how far off, all 40 points
    r = (y - model(x, a, tau, b)) / sigma
    loss = float((r ** 2).sum())
    asked.append((a, tau, b, loss))       # Minuit asked about this one
    return loss
# endregion


def run_minuit():
    # region: mn-migrad
    m = Minuit(chi2, a=1.0, tau=1.0, b=0.0)  # the start
    m.errordef = Minuit.LEAST_SQUARES     # chi-square: error at +1
    m.migrad()                            # find the minimum
    n = len(asked)                        # calls it needed
    m.hesse()                             # curvature -> error bars
    # endregion
    return m, n


def gradient(p, idx):
    """The mean of the chi-square terms of points ``idx``, and its gradient."""
    a, tau, b = p
    e = np.exp(-x[idx] / tau)
    r = (y[idx] - (a * e + b)) / sigma
    g = np.array([(-2 * r * e).mean(), (-2 * r * a * e * x[idx] / tau ** 2).mean(),
                  (-2 * r).mean()]) / sigma
    return g, float((r ** 2).mean())


def run_sgd(steps=3000, batch=8, lr=0.005, seed=7):
    pick = np.random.default_rng(seed)
    p = np.array([1.0, 1.0, 0.0])         # the same start
    path = [p.copy()]
    for _ in range(steps):
        # region: mn-sgd
        idx = pick.choice(40, batch,      # 8 of the 40 points
                          replace=False)
        g, _ = gradient(p, idx)           # their slope only: noisy
        p = p - lr * g                    # a small step downhill
        # endregion
        path.append(p.copy())
    return np.array(path)


if __name__ == "__main__":
    m, n = run_minuit()
    print("MIGRAD:", n, "calls; HESSE:", len(asked) - n, "more")
    print("minimum:", m.values, "errors:", m.errors, "chi2:", m.fval)
    path = run_sgd()
    print("SGD end:", path[-1], "mean of the last 1000:", path[-1000:].mean(0))

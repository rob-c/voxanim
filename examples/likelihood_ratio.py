"""A classifier learns the likelihood ratio, and reweights a simulation: the program of the two
likelihood-ratio animations of "Blocks and code".

    python examples/likelihood_ratio.py       # needs numpy and torch

Everything here is made up, with fixed seeds: no data are downloaded.

1. Two hypotheses about one measured value x. H0 (background): x is Gaussian, centred at 0 with
   a spread of 1. H1 (signal): centred at 1 with a spread of 0.5, narrower. The likelihood
   ratio p1(x) / p0(x) is known exactly; since the spreads differ, its log is a parabola, not a
   straight line: it is largest near x = 1.33 and falls on both sides. A small network is trained
   on equal numbers of each, with binary cross-entropy; its output s, the chance of H1, gives
   s / (1 - s), which should follow the exact ratio wherever there are events to learn from.
   Its power at a 5% false-alarm rate is measured on fresh events.
2. Reweighting. Made-up "simulation" (two measured values x and y, independent, centred at 0,
   spread 1) and "data" (x shifted by 0.4, y wider, 1.25, and x and y correlated, 0.4). The same
   kind of network is trained to tell data from simulation; each simulated event gets the
   weight s / (1 - s). The weighted simulation should look like the data in x and y, and also
   in x + y, a combination the network was never given as a column of its own.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: lr-imports
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# endregion

torch.manual_seed(0)


# region: lr-model
def classifier(inputs):                   # a small MLP
    return nn.Sequential(
        nn.Linear(inputs, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1))                 # one number: a logit
bce = nn.BCEWithLogitsLoss()              # binary cross-entropy
# endregion


# region: lr-train
def train(net, X, Y, steps):
    opt = Adam(net.parameters(), lr=3e-4)
    for _ in range(steps):
        # region: lr-step
        i = torch.randint(len(X), (512,))  # a batch of 512
        loss = bce(net(X[i]), Y[i])       # how wrong, on average
        opt.zero_grad()
        loss.backward()
        opt.step()
        # endregion
# endregion


def gauss(x, mu, sd):
    """A Gaussian's density, without its 1 / sqrt(2 pi), which cancels in a ratio."""
    z = (x - mu) / sd
    return torch.exp(-z * z / 2) / sd


def part_one():
    """A classifier learns the likelihood ratio."""
    global net, x0, x1
    # region: lr-hypotheses
    rng = np.random.default_rng(273)
    n = 20000                             # events of each kind
    x0 = rng.normal(0.0, 1.0, (n, 1))     # H0: wide, at 0
    x1 = rng.normal(1.0, 0.5, (n, 1))     # H1: narrow, at 1
    X = torch.tensor(np.r_[x0, x1]).float()
    Y = torch.zeros(2 * n, 1)
    Y[n:] = 1                             # the label: 1 for H1
    # endregion
    net = classifier(1)
    train(net, X, Y, 10000)
    # region: lr-ratio
    g = torch.linspace(-3, 5, 161)[:, None]  # x from -3 to 5
    with torch.no_grad():
        s = torch.sigmoid(net(g))         # the chance of H1
    learned = s / (1 - s)                 # the network's ratio
    exact = (gauss(g, 1.0, 0.5)           # p1(x)
             / gauss(g, 0.0, 1.0))        # / p0(x)
    # endregion
    # region: lr-power
    f0 = rng.normal(0.0, 1.0, (n, 1))     # fresh H0 events
    f1 = rng.normal(1.0, 0.5, (n, 1))     # fresh H1 events
    with torch.no_grad():
        s0 = net(torch.tensor(f0).float())
        s1 = net(torch.tensor(f1).float())
    cut = s0.quantile(0.95)               # 5% of H0 pass it
    power = (s1 > cut).float().mean()     # the share of H1 kept
    # endregion
    print("learned / exact ratio at x = 0, 1, 2:",
          (learned / exact)[[60, 80, 100], 0].numpy().round(2))
    print("power at a 5%% false-alarm rate: %.3f" % power)


def hist(v, w=None):
    """Counts (or summed weights) of ``v`` in 20 bins from -5 to 5."""
    return np.histogram(v, np.linspace(-5, 5, 21), weights=w)[0]


def part_two():
    """Reweighting simulation with a classifier."""
    global judge
    # region: lr-samples
    rng = np.random.default_rng(274)
    N = 50000                             # events of each
    sim = rng.normal(0, 1, (N, 2))        # x, y: "simulation"
    cov = [[1.0, 0.5], [0.5, 1.5625]]     # y wider; x, y tied
    data = rng.multivariate_normal(
        [0.4, 0.0], cov, N)               # "data": x shifted
    # endregion
    # region: lr-judge
    both = np.r_[sim, data]
    X = torch.tensor(both).float()
    Y = torch.zeros(2 * N, 1)
    Y[N:] = 1                             # the label: 1 for data
    judge = classifier(2)
    train(judge, X, Y, 10000)
    # endregion
    # region: lr-weights
    new = rng.normal(0, 1, (N, 2))        # fresh simulation
    with torch.no_grad():
        z = judge(torch.tensor(new).float())
    s = torch.sigmoid(z)                  # the chance of data
    w = (s / (1 - s)).numpy()[:, 0]       # one weight per event
    # endregion
    # region: lr-check
    seen = rng.multivariate_normal(
        [0.4, 0.0], cov, N)               # fresh data
    x_data = hist(seen[:, 0])             # the data's x
    x_sim = hist(new[:, 0], w)            # the weighted simulation's
    y_data = hist(seen[:, 1])
    y_sim = hist(new[:, 1], w)
    # endregion
    # region: lr-sum
    sum_data = hist(seen[:, 0] + seen[:, 1])  # x + y: never
    sum_sim = hist(new[:, 0] + new[:, 1], w)  # a column of its own
    # endregion
    for name, d, b in (("x", x_data, x_sim), ("y", y_data, y_sim),
                       ("x + y", sum_data, sum_sim)):
        print(name, "data", d[8:12], "weighted simulation", b[8:12].round())
    # region: lr-neff
    n_eff = w.sum() ** 2 / (w ** 2).sum()  # events it is worth
    big = w.max()                         # the largest weight
    # endregion
    print("effective events %.0f of %d; largest weight %.1f" % (n_eff, N, big))


if __name__ == "__main__":
    part_one()
    part_two()

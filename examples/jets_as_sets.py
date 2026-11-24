"""Jets as sets of particles: the program of the "Jets as sets of particles" animation of
"Blocks and code".

    python examples/jets_as_sets.py          # needs numpy and torch

Made-up jets (pseudo-data, seeded). A jet is a set of 6 to 20 particles, each with three
numbers: its momentum fraction z (the share of the jet's momentum it carries; a jet's z add
up to 1) and its offsets in eta and phi from the jet's axis. The z come from a Dirichlet(0.7)
(a few hard particles, many soft ones). Two kinds, half and half:

* one prong, a toy QCD jet: every particle round one core, spread 0.1 to 0.35;
* two prongs, a toy W (a W -> two quarks): two cores 0.25 to 1.0 apart, sharing the momentum
  f : 1 - f with f from 0.25 to 0.75, each core spread 0.1.

Every jet is then centred (its z-weighted centre at 0) and turned so that its spread runs
along eta, its hardest particle on the + side, as jet analyses often do. It becomes a list of
rows sorted by z, hardest first, padded with zeros to 20 rows, and a mask marks the real rows.

1. An MLP reads the 20 x 3 = 60 numbers in a row, in that sorted order.
2. A Deep Sets network applies one small network, phi, to each particle on its own, adds the
   results over the real particles (the mask), and reads the sum with a second, rho. A sum
   does not depend on the order of what it adds.

Both are trained the same way (Adam, 6,000 steps of 64 jets, on 40,000 jets); then the real
particles of each of 4,000 held-out jets are shuffled, and both are asked again.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen, lines kept
to 41 characters.
"""
# region: js-imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# endregion

MAX = 20                                  # rows: the most particles in a jet
N_TRAIN, N_TEST = 40_000, 4_000
STEPS, BATCH, LR = 6000, 64, 1e-3
bce = F.binary_cross_entropy_with_logits
torch.manual_seed(0)                      # the models' starting weights, below


def one_core(rng, n):
    """A toy QCD jet's offsets: every particle round one core."""
    return rng.normal(0.0, rng.uniform(0.1, 0.35), (n, 2))


def two_cores(rng, z):
    """A toy W's offsets: two cores ``d`` apart sharing the momentum ``f : 1 - f`` (``z`` is
    rescaled in place so that each core carries its share)."""
    n = len(z)
    f, d = rng.uniform(0.25, 0.75), rng.uniform(0.25, 1.0)
    a = rng.uniform(0.0, 2 * np.pi)
    way = np.array([np.cos(a), np.sin(a)])
    first = rng.random(n) < 0.5
    first[0], first[1] = True, False      # at least one particle in each core
    z[first] *= f / z[first].sum()
    z[~first] *= (1 - f) / z[~first].sum()
    centre = np.where(first[:, None], (1 - f) * d * way, -f * d * way)
    return centre + rng.normal(0.0, 0.1, (n, 2))


# region: js-jet
def make_jet(rng, two_prong):
    n = int(rng.integers(6, MAX + 1))     # 6 to 20 particles
    z = rng.dirichlet(np.full(n, 0.7))    # momentum fractions: sum 1
    if two_prong:                         # a toy W: two cores
        xy = two_cores(rng, z)
    else:                                 # a toy QCD jet: one core
        xy = one_core(rng, n)
    return np.c_[z, xy]                   # one row per particle
# endregion


def turn(p):
    """Centre the jet (z-weighted) and turn it so its spread runs along eta, the hardest
    particle on the + side."""
    z, xy = p[:, 0], p[:, 1:]
    xy = xy - z @ xy
    spread = (z[:, None] * xy).T @ xy
    v = np.linalg.eigh(spread)[1][:, -1]  # the long axis
    if xy[np.argmax(z)] @ v < 0:
        v = -v
    return np.c_[z, xy @ np.array([v, [-v[1], v[0]]]).T]


# region: js-pad
def as_list(p):
    p = p[np.argsort(-p[:, 0])]           # sorted by z, hardest first
    x = np.zeros((MAX, 3), np.float32)    # 20 rows of 3 numbers
    x[:len(p)] = p                        # the real ones; the rest 0
    mask = np.arange(MAX) < len(p)        # True: a real particle
    return x, mask
# endregion


def make_jets(n, seed):
    """``(X [n, 20, 3], M [n, 20], y [n])``: ``n`` made-up jets, y = 1 for two prongs."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, MAX, 3), np.float32)
    M = np.zeros((n, MAX), bool)
    y = np.zeros(n, np.float32)
    for i in range(n):
        two = rng.random() < 0.5
        X[i], M[i] = as_list(turn(make_jet(rng, two)))
        y[i] = two
    return X, M, y


# region: js-shuffle
def shuffled(X, M, rng):
    Xs = X.copy()
    for x, m in zip(Xs, M):
        n = m.sum()                       # the real rows only
        x[:n] = x[rng.permutation(n)]     # same particles, new order
    return Xs
# endregion


# region: js-mlp
mlp = nn.Sequential(nn.Flatten(),         # 20 x 3 -> 60 in a row
    nn.Linear(60, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1))
# endregion

# region: js-deepsets
phi = nn.Sequential(                      # each particle: 3 -> 32
    nn.Linear(3, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU())
rho = nn.Sequential(                      # the jet's sum: 32 -> 1
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1))
# endregion


# region: js-sum
def deep_sets(x, mask):
    h = phi(x)                            # [B, 20, 32]: one row each
    h = h * mask[..., None]               # the pads' rows -> 0
    return rho(h.sum(1))                  # add them up, then rho
# endregion


def flat_mlp(x, mask):
    return mlp(x)                         # the pads are read as zeros


def train(model, params, X, M, y):
    opt = torch.optim.Adam(params, lr=LR)
    pick = np.random.default_rng(5)       # the same batches for both
    for _ in range(STEPS):
        i = torch.as_tensor(pick.integers(0, len(X), BATCH))
        # region: js-step
        logit = model(X[i], M[i])[:, 0]
        loss = bce(logit, y[i])           # how wrong, on 64 jets
        opt.zero_grad()
        loss.backward()
        opt.step()
        # endregion


def measure(model, X, M, y, Xs):
    # region: js-measure
    with torch.no_grad():
        a = torch.sigmoid(model(X, M))[:, 0]   # sorted
        b = torch.sigmoid(model(Xs, M))[:, 0]  # shuffled
    right = ((a > 0.5) == y).float().mean()
    moved = (a - b).abs()                 # how far each answer moved
    # endregion
    after = ((b > 0.5) == y).float().mean()
    return float(right), float(after), moved


if __name__ == "__main__":
    X, M, y = make_jets(N_TRAIN, 1)
    Xt, Mt, yt = make_jets(N_TEST, 2)
    Xs = shuffled(Xt, Mt, np.random.default_rng(3))
    # region: js-again
    k = 0                                 # any held-out jet
    p = Xs[k][Mt[k]]                      # its shuffled particles
    x, mask = as_list(p)                  # sorted again
    print((x == Xt[k]).all())             # True: the same list
    # endregion
    T = torch.as_tensor
    X, M, y, Xt, Mt, yt, Xs = map(T, (X, M, y, Xt, Mt, yt, Xs))
    train(flat_mlp, mlp.parameters(), X, M, y)
    train(deep_sets, [*phi.parameters(), *rho.parameters()], X, M, y)
    for name, model in (("MLP", flat_mlp), ("Deep Sets", deep_sets)):
        right, after, moved = measure(model, Xt, Mt, yt, Xs)
        print("%-9s accuracy %.4f, shuffled %.4f; answers moved: mean %.3g, largest %.3g"
              % (name, right, after, moved.mean(), moved.max()))

"""Event weights and scale factors: the program of the "From simulation to a result" animations
of "Blocks and code" (event weights; simulation is not data: scale factors).

    python examples/weights_and_scale_factors.py          # needs numpy

Everything here is made up, with fixed seeds. One made-up variable x (think of it as a
classifier's input) is drawn for three simulated samples:

    signal   x ~ Normal(1.00, 0.10)   cross-section  50 fb   20,000 events generated
    bkg 1    x ~ Normal(0.60, 0.12)   cross-section 200 fb   50,000 events generated
    bkg 2    x ~ Normal(0.75, 0.12)   cross-section 120 fb   10,000 events generated

and a luminosity of 100 fb^-1.

1. Event weights. Each simulated event counts for sigma x L / N real events (N: the events
   generated for its sample). Weighted, each sample's histogram adds up to sigma x L, the
   number of events expected. The weights go into the loss (each event's cross-entropy times
   its weight; a one-variable logistic regression is trained with and without them), into an
   efficiency (the weighted share that passes a cut), and some generators make negative
   weights: a version of bkg 2 where 2,000 of its 10,000 events count negative, and what that
   does to its effective number of events, (sum w)^2 / sum w^2.
2. Scale factors. The cut is chosen on simulation (it keeps 80% of the simulated signal).
   The made-up "data" measure signal-like objects with a wider spread (0.13, not 0.10), so the
   cut keeps fewer of them. In a control region of objects known to be signal-like (as tag and
   probe gives), the efficiency is counted in simulation and in data; the scale factor is
   their ratio, with its statistical error, and it corrects the simulation's prediction in the
   signal region (x above the cut), which is then compared with the data.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: ws-imports
import numpy as np

# endregion

SHAPES = {"signal": (1.00, 0.10),         # x: mean and spread
          "bkg 1": (0.60, 0.12),
          "bkg 2": (0.75, 0.12)}
DATA_SPREAD = 0.13                        # made up: the data's signal is wider

# region: ws-samples
L = 100.0                                 # luminosity, fb^-1
samples = {                               # cross-section (fb), events made
    "signal": (50.0, 20000),
    "bkg 1": (200.0, 50000),
    "bkg 2": (120.0, 10000)}
# endregion

rng = np.random.default_rng(1)
x = {name: rng.normal(*SHAPES[name], n) for name, (_, n) in samples.items()}

# region: ws-weight
w = {}
for name, (sigma, n) in samples.items():
    w[name] = np.full(n, sigma * L / n)   # what each one counts for
# endregion

# region: ws-hist
edges = np.linspace(0.0, 1.6, 17)         # 16 bins of x
for name in samples:
    h, _ = np.histogram(x[name], edges,
                        weights=w[name])
    print(name, round(h.sum()))           # = cross-section x L
# endregion


def events(weighted):
    """All the simulated events together: x, label (1: signal), weight (all 1 if not)."""
    xs = np.concatenate([x[k] for k in samples])
    ys = np.concatenate([np.full(len(x[k]), float(k == "signal")) for k in samples])
    ws = np.concatenate([w[k] for k in samples])
    return xs, ys, ws if weighted else np.ones_like(ws)


def train(weighted, steps=3000, lr=10.0):
    """A logistic regression on x, score s = 1 / (1 + exp(-(a x + b))), by gradient descent
    on the (weighted) mean cross-entropy over every event. Returns a, b and the losses."""
    x, y, w = events(weighted)
    a, b, losses = 0.0, 0.0, []
    for _ in range(steps):
        # region: ws-loss
        s = 1 / (1 + np.exp(-(a * x + b)))  # the score
        ce = -(y * np.log(s)
               + (1 - y) * np.log(1 - s))
        loss = (w * ce).sum() / w.sum()   # each counts w times
        g = w * (s - y) / w.sum()         # the loss's slope
        a -= lr * (g * x).sum()
        b -= lr * g.sum()
        # endregion
        losses.append(loss)
    return a, b, np.array(losses)


# region: ws-cut
cut = np.quantile(x["signal"], 0.2)       # keeps 80% of signal
# endregion

xb = np.concatenate([x["bkg 1"], x["bkg 2"]])     # the two backgrounds together
wb = np.concatenate([w["bkg 1"], w["bkg 2"]])

# region: ws-eff
keep = xb > cut                           # passes the cut
print(keep.mean())                        # counts: the wrong mix
print(wb[keep].sum() / wb.sum())          # weighted: the right mix
# endregion


def negative():
    """bkg 2 again, as a generator that makes negative weights would give it: the same events,
    2,000 of the 10,000 counting negative. Returns the signs, the weights and N_eff."""
    rng = np.random.default_rng(3)
    sigma, n = samples["bkg 2"]
    # region: ws-negative
    sign = np.ones(n)
    neg = rng.choice(n, 2000, replace=False)
    sign[neg] = -1                        # 2,000 count negative
    wn = sign * sigma * L / sign.sum()    # +2 or -2 each
    n_eff = wn.sum() ** 2 / (wn ** 2).sum()
    # endregion
    return sign, wn, n_eff


# made-up data: a control region of signal-like objects (probes), in simulation and in data,
# and the data themselves: each process's expected count, Poisson-fluctuated
rng = np.random.default_rng(2)            # the made-up data's own seed
# region: ws-probes
probe_mc = rng.normal(1.0, 0.10, 20000)   # simulation
probe_data = rng.normal(1.0, 0.13, 5000)  # data: wider
# endregion
n_mc, n_data = len(probe_mc), len(probe_data)
data = np.concatenate([
    rng.normal(SHAPES[k][0], DATA_SPREAD if k == "signal" else SHAPES[k][1],
               rng.poisson(samples[k][0] * L)) for k in samples])

# region: ws-eff-cr
eff_mc = (probe_mc > cut).mean()          # simulation
eff_data = (probe_data > cut).mean()      # data
# endregion

# region: ws-sf
sf = eff_data / eff_mc                    # the scale factor
err = sf * np.sqrt(                       # its statistical error
    (1 - eff_data) / (eff_data * n_data)
    + (1 - eff_mc) / (eff_mc * n_mc))
# endregion


# region: ws-passing
def passing(name):                        # weighted, above the cut
    return w[name][x[name] > cut].sum()
# endregion


# region: ws-correct
sig = passing("signal")
bkg = passing("bkg 1") + passing("bkg 2")
predicted = sig + bkg                     # simulation as it is
corrected = sf * sig + bkg                # its signal scaled
observed = (data > cut).sum()             # the data
# endregion


if __name__ == "__main__":
    for weighted in (False, True):
        a, b, losses = train(weighted)
        s = 1 / (1 + np.exp(-(a * events(weighted)[0] + b)))
        ww = events(weighted)[2]
        print("weighted" if weighted else "unweighted", "a, b:", a, b, "loss:", losses[-1],
              "mean score:", (ww * s).sum() / ww.sum(), "s = 0.5 at x =", -b / a)
    sign, wn, n_eff = negative()
    print("negative weights:", (sign < 0).sum(), "of", len(sign), "sum:", wn.sum(),
          "N_eff:", n_eff)
    print("cut:", cut, "efficiency: simulation", eff_mc, "data", eff_data)
    print("scale factor:", sf, "+-", err)
    print("signal region: predicted", predicted, "corrected", corrected, "observed", observed)

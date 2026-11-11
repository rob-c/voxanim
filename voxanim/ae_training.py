"""An autoencoder learning its latent space.

The untrained network first: a digit goes through the random weights, its
code lands in the latent space among the 600 test codes, all in one clump,
and the output is a blur.  Then training runs, on a log time axis (every
tenfold in steps takes as long): the test codes drift and sort themselves,
the three digits of the chain leave trails, the chain shows the live
activations and the reconstruction sharpening, and a loss curve draws
itself.  The last frame is the trained network of ``autoencoder-dense-2d``.

Every number comes from one recorded training run
(:func:`voxanim.autoencoder.training_run`); between two snapshots the
screen interpolates linearly.  The latent box follows the cloud: its axes are
the robust axes of :class:`~voxanim.ae_anim.AutoencoderAnimation` recomputed
for the codes of each snapshot, so the sorting stays visible while the codes
grow fivefold, and the numbers at the axis ends show the scale.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .ae_anim import CODE, DIGIT_COLORS, FRAME, AutoencoderAnimation, _shapes, chain_colors
from .autoencoder import AETrace
from .easing import ease_in_out, remap, smoothstep
from .netanim import div_colors

CURVE = ((0.80, 0.95), (0.23, 0.37))     # the loss curve's box ((x0, x1), (top, bottom)), layout
SPAN = 24.0                               # seconds the training phase takes
SWITCH = 0.03                             # of the phase: the chain refilling with the next digit
TRAIL = (0.55, 0.55, 0.55)                # colour of the first, oldest part of a trail


class AutoencoderTraining(AutoencoderAnimation):
    """One digit through the untrained autoencoder, then its training run.

    ``run`` is a :class:`voxanim.autoencoder.AERun`.  The opening reuses the
    base class's digit choreography on the untrained network; the training
    phase is new.
    """

    PHASES = ("untrained", "training", "trained")

    def __init__(self, run):
        self.run = run
        shapes = _shapes(run.kind)
        k = len(run.images)

        def fwd(s, d):
            """A forward dict (as :func:`voxanim.autoencoder.forward`, batch
            removed) of digit ``d`` at snapshot ``s``, from the recorded activations."""
            acts = [np.asarray(run.acts[j][s, d], np.float64).reshape(shapes[j] or (run.latent,))
                    for j in range(len(shapes))]
            return dict(acts=acts, z=acts[CODE])

        untrained = AETrace(run.kind, run.latent, run.P0, run.images, run.labels,
                            [fwd(0, d) for d in range(k)], run.cloud[0], run.cloud_labels)
        super().__init__(untrained, title="Autoencoder: learning the latent space")
        # the latent axes (median, spread, box range) at every snapshot
        ax = []
        for cloud in run.cloud:
            lo, hi = self._axes(cloud)
            ax.append([self.med, self.mad, lo, hi])
        self.ax = np.array(ax)                                     # (S, 4, L)
        self.S = len(run.steps)
        self.log_loss = (np.log10(run.losses.min()) - 0.05, np.log10(run.losses.max()) + 0.05)

        # ---- timeline: the cloud, one digit through the untrained network, training
        Tm, t = {}, 1.6
        segs = []
        for d in (1.2, 1.0, 0.9, 1.2, 0.9, 1.0, 1.2, 1.6):
            segs.append((t, t + d))
            t += d
        self.digit_t = [segs]                   # the base class's choreography, digit 0 only
        t += 0.6
        Tm["train"] = (t, t + SPAN)
        t += SPAN + 0.5
        Tm["done"] = (t, t + 2.0)
        t += 2.0
        self.Tm = Tm
        self.finish(t)

    # ------------------------------------------------------------------ the run
    def step_of(self, t):
        """0 untrained, 1 training, 2 trained."""
        if t >= self.Tm["train"][1]:
            return 2
        return 1 if t >= self.Tm["train"][0] else 0

    def snapshot(self, pos):
        """The run's state at fractional snapshot index ``pos`` (linear between
        snapshots).  Sets the latent axes; returns ``cloud`` ``(m, L)``, ``acts``
        (per chain tensor, ``(k, values)``), ``loss`` and ``step``."""
        run = self.run
        i = min(int(pos), self.S - 2)
        f = pos - i

        def lerp(a):
            return a[i] + (a[i + 1] - a[i]) * f

        self.med, self.mad, self.lo, self.hi = lerp(self.ax)
        return dict(cloud=lerp(run.cloud), acts=[lerp(a) for a in run.acts],
                    loss=float(lerp(run.losses)), step=float(lerp(run.steps)), i=i)

    def unwarp(self, w):
        """Latent values at warped coordinates ``w``: the inverse of :meth:`_warp`."""
        return self.med + 2.0 * self.mad * np.sinh(w)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        u = self.progress("train", t)
        run, k = self.run, len(self.run.images)
        if u <= 0:
            self.snapshot(0.0)
            landed, subtitle = self._digit(sc, t, 0, back, active=t >= self.digit_t[0][0][0])
        else:
            pos = (self.S - 1) * smoothstep(0.0, 1.0, u)
            snap = self.snapshot(pos)
            self.cloud_pts = self.to_box(snap["cloud"])
            d = min(k - 1, int(u * k))                 # the digit in the chain: each in turn
            refill = 1.0 if d == 0 else remap(u, d / k, d / k + SWITCH)
            self._chain(snap, d, refill, back)
            self._trails(sc, snap, d, back)
            self._loss_curve(sc, snap, back)
            landed = {int(lab) for lab in run.labels}
            subtitle = self._training_subtitle(u, snap)
        self._cloud(sc, t, back, landed)
        self._scale(sc, t, back)
        if t < self.digit_t[0][0][0] or reset > 0:          # the loop returns to the opening
            subtitle = ("before training: random weights scatter the %d test codes at random; "
                        "no digit has a place of its own" % len(run.cloud_labels))
        self.subtitle = subtitle
        self.step_indicator(sc, ph, self.PHASES)
        self._captions(sc, back)

    def _training_subtitle(self, u, snap):
        run = self.run
        step, epoch = int(round(snap["step"])), int(snap["step"] // run.steps_per_epoch) + 1
        epochs = run.steps[-1] // run.steps_per_epoch
        if u >= 1:
            return ("after %d epochs: loss %.4f; the digits sort themselves by shape, "
                    "though the network never sees a label" % (epochs, snap["loss"]))
        if u < 0.35:
            return ("training, step %d of %d: the loss falls fast and the codes spread out "
                    "(watch the axis scale grow)" % (step, run.steps[-1]))
        if u < 0.7:
            return ("training, step %d (epoch %d): digits of a kind drift together, though "
                    "the network never sees a label" % (step, epoch))
        return ("training, step %d (epoch %d of %d): the slow part: the clusters tighten "
                "and the output sharpens" % (step, epoch, epochs))

    # -- the chain, live --------------------------------------------------------------------
    def _chain(self, snap, d, refill, back):
        """The chain showing digit ``d`` at this snapshot; ``refill`` < 1 while it
        is being fed in (the layers after the input fill row by row)."""
        for j, b in enumerate(self.blocks):
            if b is None:
                continue
            a = snap["acts"][j][d]
            cols = chain_colors(j, a)
            shp = self.shapes[j]
            self._fill(j, (cols.reshape(shp + (3,)), a.reshape(shp)), 1.0 if j == 0 else refill,
                       back)
        z = snap["acts"][CODE][d]
        self.code.base[0, :, 0] = div_colors(z / np.abs(snap["cloud"]).max(0), 1.0)
        self.code.present[...] = True
        self.code.opacity *= back

    # -- the tracked digits' codes ----------------------------------------------------------
    def _trails(self, sc, snap, d, back):
        """Each chain digit's code: a large marker, and the trail of where it
        has been (older is greyer); the digit in the chain gets a frame."""
        run, i = self.run, snap["i"]
        for j in range(len(run.images)):
            col = DIGIT_COLORS[int(run.labels[j])]
            path = self.to_box(np.vstack([run.acts[CODE][:i + 1, j], snap["acts"][CODE][j]]))
            n = len(path)
            for s in range(n - 1):
                w = (s + 1) / n
                c = np.asarray(TRAIL) + (np.asarray(col) - TRAIL) * w
                sc.line(path[s], path[s + 1], width=1.3, color=tuple(c),
                        opacity=(0.25 + 0.65 * w) * back, bias=0.05)
            p = path[-1]
            sc.voxel(p, col, back, glow=0.35, shadow=False, reflect=False, size=1.5)
            if j == d:
                sc.wire_box(p - 1.0, p + 1.0, color=FRAME, width=1.2, opacity=0.9 * back, bias=0.1)
            sc.probe(("code", j), p, tol=4.0)

    # -- the box's scale -------------------------------------------------------------------
    def _scale(self, sc, t, back):
        """The latent values at the ends of the plane's axes (2-D only): the
        box follows the cloud, and these numbers say how far it has grown.
        They fade in with the cloud."""
        if self.L != 2:
            return
        op = smoothstep(0.1, 1.4, t) * back * self._cap
        z_lo, z_hi = self.unwarp(self.lo), self.unwarp(self.hi)
        c, a, b = self.centre, self.e1 / 2, self.e2 / 2
        cam = self.cam
        marks = [(c - a - b, "%.1f" % z_lo[0], (0.0, 0.012), "lt"),
                 (c + a - b, "%.1f" % z_hi[0], (0.0, 0.012), "rt"),
                 (c - a - b, "%.1f" % z_lo[1], (-0.012, 0.0), "rb"),
                 (c - a + b, "%.1f" % z_hi[1], (-0.012, 0.0), "rt")]
        for p, s, (dx, dy), anchor in marks:
            q = cam.project(p)
            sc.text(q[0] / cam.width + dx, q[1] / cam.height + dy, s, 0.026, "mono",
                    layout.NOTE_INK, op, anchor)

    # -- the loss curve ------------------------------------------------------------------------
    def _loss_curve(self, sc, snap, back):
        """Test loss against training step (log-log; step 0 at the origin), drawn
        snapshot by snapshot as training runs."""
        run, cam, txt = self.run, self.cam, self.fr
        op = back * self._cap
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = CURVE
        lo, hi = self.log_loss
        total = run.steps[-1]

        def fx(step):
            return x0 + (x1 - x0) * np.log10(step + 1) / np.log10(total + 1)

        def pt(step, loss):
            fy = y1 - (y1 - y0) * (np.log10(loss) - lo) / (hi - lo)
            return cam.frac(fx(step), fy)

        axis = (0.5, 0.52, 0.54)
        sc.line(cam.frac(x0, y1), cam.frac(x1, y1), width=1.0, color=axis, opacity=0.7 * op,
                bias=0.0)
        sc.line(cam.frac(x0, y1), cam.frac(x0, y0), width=1.0, color=axis, opacity=0.7 * op,
                bias=0.0)
        sc.text(*txt(x0, y0 - 0.014), "loss", 0.026, "regular", layout.CAPTION_INK, op, "lb")
        for step in (1, 10, 100, 1000):
            sc.text(*txt(fx(step), y1 + 0.01), str(step), 0.024, "regular", layout.NOTE_INK, op,
                    "mt")
        sc.text(*txt((x0 + x1) / 2, y1 + 0.046), "training steps", 0.026, "regular",
                layout.CAPTION_INK, op, "mt")
        prev = None
        for step, loss in [*zip(run.steps[:snap["i"] + 1], run.losses[:snap["i"] + 1]),
                           (snap["step"], snap["loss"])]:
            p = pt(step, loss)
            if prev is not None:
                sc.line(prev, p, width=1.2, color=(0.75, 0.22, 0.22), opacity=0.85 * op,
                        bias=0.0)
            prev = p
        sc.voxel(prev, (0.75, 0.22, 0.22), op, glow=0.4, shadow=False, reflect=False, size=0.7)

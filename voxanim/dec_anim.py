"""Deep Embedded Clustering (DEC) and Improved DEC (IDEC), animated from a
real training run (:func:`voxanim.deep_models.cluster_train`).

    latent space (3-D codes of 800 test digits + 10 cluster centres)   q vs p bars
                                                                        accuracy
    reconstructions of four digits through the decoder

1. the pretrained autoencoder's codes, coloured by their true label,
2. k-means drops 10 centres; each code is coloured by the cluster it
   belongs to (matched to the digit that cluster mostly holds),
3. training: the soft assignment q (Student-t kernel to each centre) is
   pushed towards the sharpened target p = q^2 / f, i.e. minimise
   KL(P || Q): codes pull into their confident clusters and the centres move;
   accuracy rises,
4. DEC drops the decoder, so the codes drift away from anything it can draw;
   IDEC keeps it and adds the reconstruction loss, and its reconstructions
   stay sharp.
"""

from __future__ import annotations

import numpy as np

from . import deep_models as D
from . import layout
from .ae_anim import DIGIT_COLORS
from .easing import ease_in_out, remap, smoothstep
from .netanim import NetAnimation, gray_colors
from .tensor import Block


class ClusterAnimation(NetAnimation):
    PHASES = ("embedding", "k-means", "training", "result")

    def __init__(self, run):
        idec = run.kind == "idec"
        title = "IDEC: clustering + reconstruction" if idec else \
            "DEC: deep embedded clustering"
        super().__init__(title, "", 9.0)
        self.run, self.idec = run, idec
        self.snaps = run.snaps
        cam = self.cam
        allz = np.concatenate([s["z"] for s in self.snaps])
        self.lo = np.percentile(allz, 1, axis=0)
        self.hi = np.percentile(allz, 99, axis=0)
        S = 20.0
        c = cam.frac(0.38, 0.46)
        self.box_lo, self.box_hi = c - S / 2, c + S / 2
        # colour of every cluster: the digit it ends up matching
        self.maps = [D.cluster_accuracy(s["q"].argmax(1), run.cloud_labels)[1]
                     for s in self.snaps]
        self.focus = int(np.argmax(self.snaps[0]["q"].max(1) < 0.4))   # an unsure digit
        self.rec_err = [float(np.mean((s["recon"] - run.show) ** 2)) for s in self.snaps]
        # q / p bars and reconstructions
        self.qbar = Block((8, 10, 1), scale=0.9, name="q")
        self.qbar.place(cam.frac(0.80, 0.36))
        self.pbar = Block((8, 10, 1), scale=0.9, name="p")
        self.pbar.place(cam.frac(0.80, 0.56))
        self.recs = [self.sheet((28, 28), (0.69 + 0.075 * k, 0.80), 0.17, "r%d" % k)
                     for k in range(4)]
        self.blocks_extra = [self.qbar, self.pbar]
        self.hold = 2.5
        self.Tm = dict(embed=(0.3, 1.8), kmeans=(1.8, 4.0), switch=(4.0, 5.4),
                       train=(5.6, 27.0))
        self.finish(27.4)

    def to_box(self, z):
        u = np.clip((np.asarray(z) - self.lo) / (self.hi - self.lo), 0.02, 0.98)
        return self.box_lo + u * (self.box_hi - self.box_lo)

    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def snap_at(self, t, reset):
        """Interpolated snapshot index (float) for time ``t``."""
        u = self.p_("train", t) * (1.0 - ease_in_out(reset))
        return u * (len(self.snaps) - 1)

    def step_of(self, t):
        if t >= self.Tm["train"][1]:
            return 3
        if t >= self.Tm["switch"][1]:
            return 2
        if t >= self.Tm["kmeans"][0]:
            return 1
        return 0

    def draw(self, sc, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        x = self.snap_at(t, reset)
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(self.snaps) - 1)
        w = ease_in_out(x - i0)
        s0, s1 = self.snaps[i0], self.snaps[i1]
        z = s0["z"] + (s1["z"] - s0["z"]) * w
        mu = s0["mu"] + (s1["mu"] - s0["mu"]) * w
        q = s0["q"] + (s1["q"] - s0["q"]) * w
        snap = self.snaps[int(round(x))]
        acc = s0["acc"] + (s1["acc"] - s0["acc"]) * w
        mapping = self.maps[int(round(x))]
        labels = self.run.cloud_labels

        fade = smoothstep(0.0, 1.0, self.p_("embed", t)) * back    # fades in, and out again
        cl = self.p_("kmeans", t) * (1 - smoothstep(0.0, 0.4, reset))
        sw = self.p_("switch", t) * (1 - smoothstep(0.0, 0.5, reset))
        # the cube and the codes
        sc.wire_box(self.box_lo, self.box_hi, color=layout.NOTE_INK, width=1.0,
                    opacity=0.75, bias=0.0)
        assign = q.argmax(1)
        for n in range(len(z)):
            true_c = DIGIT_COLORS[labels[n]]
            clus_c = DIGIT_COLORS[mapping[assign[n]]]
            col = true_c + (clus_c - true_c) * sw
            sc.voxel(self.to_box(z[n]), col, fade, shadow=False, reflect=False, size=0.45)
        # the centres
        for j in range(D.K):
            if cl <= j / D.K * 0.8:
                continue
            p = self.to_box(mu[j])
            col = DIGIT_COLORS[mapping[j]] if sw > 0 else (0.95, 0.95, 0.95)
            sc.voxel(p, col, 1.0, glow=0.35, shadow=False, reflect=False, size=1.6)
            sc.wire_box(p - 1.2, p + 1.2, color=(0.97, 0.97, 0.96), width=1.3, opacity=0.9,
                        bias=0.1)
        # the focus digit
        pf = self.to_box(z[self.focus])
        if sw > 0:
            sc.wire_box(pf - 0.9, pf + 0.9, color=(0.12, 0.36, 0.66), width=1.6,
                        opacity=sw, bias=0.1)
        # q and p for the focus digit
        qf = q[self.focus]
        pt = D.target(q)[self.focus]
        for blk, vals in ((self.qbar, qf), (self.pbar, pt)):
            blk.present[...] = False
            h = np.clip(np.round(vals * 8).astype(int), 0, 8)
            h[vals > 0.03] = np.maximum(h[vals > 0.03], 1)
            for j in range(D.K):
                blk.present[8 - h[j]:, j, 0] = sw > 0
                blk.base[:, j, 0] = DIGIT_COLORS[mapping[j]]
            blk.opacity *= sw
        # reconstructions
        for k, sh in enumerate(self.recs):
            r = snap["recon"][k]
            sh.fill(gray_colors(r))
            sh.cage = 1.0
        for sh in self.recs:
            sh.fade(fade)

        self.subtitle = [
            "the pretrained autoencoder's 3-D codes, coloured by their true label",
            "k-means places 10 cluster centres; each code joins its nearest",
            ("minimise KL(P || Q): q = soft assignment to each centre, p = q sharpened. "
             "Codes pull into their confident clusters") + (
                 ";  + reconstruction loss" if self.idec else ""),
            ("IDEC: accuracy %.0f%%; reconstruction error %.3f (was %.3f): still sharp"
             if self.idec else
             "DEC: accuracy %.0f%%; reconstruction error %.3f (was %.3f): codes drifted")
            % (100 * acc, self.rec_err[-1], self.rec_err[0]),
        ][ph]
        self.step_indicator(sc, ph, self.PHASES)
        cap = self.caption
        cap(sc, (self.qbar.origin, self.qbar.origin + self.qbar.extent),
            "q: soft assignment (outlined digit)", sw * back)
        cap(sc, (self.pbar.origin, self.pbar.origin + self.pbar.extent),
            "p: sharpened target", sw * back)
        lo = self.recs[0].bbox()[0]
        hi = self.recs[-1].bbox()[1]
        cap(sc, (lo, hi), "decoder output" + (" (trained)" if self.idec else
                                              " (not trained)"), fade * back)
        it = int(round(snap["it"]))
        cap(sc, (self.box_lo, self.box_hi), "iteration %d   accuracy %.0f%%" % (it, 100 * acc)
            if sw > 0 else "", back)

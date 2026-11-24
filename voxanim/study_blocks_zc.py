"""The numbers behind animations 265-268 of "Blocks and code" (group ``zc``, "From autoencoder
to GAN"): an autoencoder's two blocks wired again as a GAN. Everything comes from
:mod:`voxanim.study_aegan`, the numpy twin of ``examples/ae_gan.py`` (one run, one seed,
MNIST pixels 0..1; ``tests/test_aegan_equivalence.py`` matches it to torch).

* ``ae-to-gan-rewire``: the encoder (``784 -> 256 -> 16``, 205,072 numbers) and the decoder
  (``16 -> 256 -> 784``, 205,840), trained together for 3,000 steps; held-out digit 0 (a 7)
  squeezed to its 16-number code and rebuilt. Rewired (region "ag-rewire"): the generator
  *is* the decoder, every number kept, fed made-up codes (noise with the mean and spread of
  real codes); the judge is the encoder's first layer and ReLU, every number kept (200,960),
  and a new last layer ``256 -> 1`` (257 numbers); the encoder's last layer (4,112 numbers)
  is left out. The new head starts random: it has learned nothing yet (:func:`rewire`).
  The weight tiles drawn are corners of the real tensors (:func:`patch`), the same before
  and after the move.
* ``ae-gan-blurry-or-sharp``: real held-out digits, their rebuilds, and the decoder fed the
  same made-up codes before and after the GAN game (the first three of each, by a rule:
  the first held-out digits and the first codes, not picked for looks); over 1,000 of each,
  the share of grey pixels (0.2 to 0.8) and the course classifier's certainty. Why rebuilds
  are grey, in exact arithmetic: a pixel that is ink in half the digits a code could be
  scores best, pixel by pixel, at 0.5 (:func:`pixel_scores`).
* ``ae-gan-rebuild-or-invent``: four held-out digits and their rebuilds, read by the
  classifier; 928 of 1,000 rebuilds read as their digit (real digits: 949); the generator's
  only input is 16 made-up numbers; 1,000 made digits counted by what the classifier reads
  them as, before and after the game; the 16 made from the fixed codes after 6,000 steps.
* ``ae-gan-one-loss-or-two``: the autoencoder's loss over its 3,000 steps (0.69 at the
  first, 0.10 over the last 100); the judge's and generator's over 6,000 steps, as means of
  100 (1.43 to 1.20, 0.67 to 0.88), against a judge that cannot tell (2 ln 2 = 1.39 and
  ln 2 = 0.69, exact); the first two fixed codes' pictures at steps 0, 500, 2,000 and 6,000.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from . import study_aegan as A
from . import study_course as S
from .blocks import code_lines, commas, val
from .study import Panel, Study

SOURCE = ("MNIST; examples/ae_gan.py run once in numpy (voxanim.study_aegan): an autoencoder "
          "784-256-16-256-784 trained 3,000 steps (Adam, lr 0.001, batches of 64, binary "
          "cross-entropy), its blocks wired as a GAN and trained 6,000 steps (Adam, lr 0.0002)")
DIGIT = 0                    # 265: the held-out digit squeezed and rebuilt (a 7)
REAL = 1                     # 265: the real digit shown to the judge (a 2)
CODE_SHOWN = 0               # 265: the made-up code fed to the generator: fixed[0]
SHOWN = 3                    # 266: the first three of each kind
PAIRS = 4                    # 267: the first four held-out digits and their rebuilds
CHUNK = 100                  # 268: the GAN's losses as means of 100 steps
SNAP_SHOWN = 2               # 268: the first two of the 16 fixed codes, at each snapshot
CODE_RANGE = 10.0            # the codes' tiles: paper at -10, full ink at +10


def pct(v, places=1):
    """``9.4%``."""
    return "%.*f%%" % (places, 100.0 * float(v))


def counts():
    """Numbers in each layer and block, from the layer sizes (``in x out + out``)."""
    lin = lambda i, o: i * o + o                         # noqa: E731
    c = dict(enc1=lin(784, 256), enc2=lin(256, 16), dec1=lin(16, 256), dec2=lin(256, 784),
             head=lin(256, 1))
    c["encoder"] = c["enc1"] + c["enc2"]
    c["decoder"] = c["dec1"] + c["dec2"]
    c["judge"] = c["enc1"] + c["head"]
    c["generator"] = c["decoder"]
    return c


def patch(W, rows, cols=5, centre=False):
    """A corner of weight matrix ``W`` ``[out, in]`` to draw as tiles, ``rows x cols``,
    magnitudes: rows evenly spread over the outputs; for a 784-input layer, inputs from the
    middle of the picture (row 14, columns 12 on), where strokes are."""
    W = np.abs(np.asarray(W, float))
    r = np.linspace(0, W.shape[0] - 1, rows).round().astype(int)
    if centre:
        c = 14 * 28 + 12 + np.arange(cols)
    else:
        c = np.linspace(0, W.shape[1] - 1, cols).round().astype(int)
    return W[np.ix_(r, c)]


# the rows of tiles drawn for each layer: its height on screen says its size of output
LAYER_ROWS = dict(enc1=12, enc2=5, dec1=12, dec2=16, head=1)


def patches(which_enc="enc", which_dec="dec", head="judge0"):
    """The tiles drawn of each layer (corners of the real weights)."""
    e, d = A.blocks(which_enc), A.blocks(which_dec)
    h = A.blocks(head)
    return dict(enc1=patch(e["0.weight"], LAYER_ROWS["enc1"], centre=True),
                enc2=patch(e["2.weight"], LAYER_ROWS["enc2"]),
                dec1=patch(d["0.weight"], LAYER_ROWS["dec1"]),
                dec2=patch(d["2.weight"].T, LAYER_ROWS["dec2"]),
                head=patch(h["2.weight"], 1, cols=5))


def grey_mask(img):
    """The pixels counted as grey (between 0.2 and 0.8)."""
    img = np.asarray(img)
    return (img > A.GREY[0]) & (img < A.GREY[1])


def pixel_scores(p=0.5, guesses=(0.1, 0.5, 0.9)):
    """Binary cross-entropy of painting ``q`` where the pixel is ink in a share ``p`` of the
    digits a code could be: ``-(p ln q + (1 - p) ln(1 - q))``, lowest at ``q = p``."""
    return [-(p * math.log(q) + (1 - p) * math.log(1 - q)) for q in guesses]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


@functools.lru_cache(maxsize=1)
def common():
    """What the four share: the counts, the measured results, the code."""
    r = A.run()
    res = A.results()
    s = A.summary()
    c = counts()
    kinds = ("real", "rebuilt", "before", "after")
    grey = {k: res[k]["grey"] for k in kinds}
    sure = {k: res[k]["sure"] for k in kinds}
    return dict(
        source=SOURCE, counts=c, counts_text={k: commas(v) for k, v in c.items()},
        kinds=list(kinds), grey=grey, grey_text={k: pct(v) for k, v in grey.items()},
        sure=sure, sure_text={k: val(v) for k, v in sure.items()},
        rebuild_right=res["rebuild_right"], real_right=res["real_right"],
        rebuild_right_text=pct(res["rebuild_right"]), real_right_text=pct(res["real_right"]),
        rebuild_right_n=int(round(res["rebuild_right"] * A.SAMPLES)),
        real_right_n=int(round(res["real_right"] * A.SAMPLES)),
        made_counts={k: res[k]["counts"] for k in kinds},
        samples=A.SAMPLES, samples_text=commas(A.SAMPLES),
        ae_steps=A.AE_STEPS, gan_steps=A.GAN_STEPS,
        ae_first=s["ae_loss"][0], ae_last=s["ae_loss"][1],
        lj_first=s["lj"][0], lj_last=s["lj"][1], lg_first=s["lg"][0], lg_last=s["lg"][1],
        code_blocks=list(code_lines("ag-blocks")), code_ae=list(code_lines("ag-ae")),
        code_rewire=list(code_lines("ag-rewire")), code_judge=list(code_lines("ag-judge")),
        code_generator=list(code_lines("ag-generator")),
        _r=r)


def _public(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def test_digits(n):
    """The first ``n`` held-out digits ``(pixels [n, 784], labels)``."""
    return A.pixels()[1][:n], S.data()[3][:n]


def rebuilt(x):
    return A.decode(A.blocks("dec"), A.encode(A.blocks("enc"), x))


def square(img):
    return np.asarray(img, float).reshape(-1, 28, 28)


# ====================================================================== 265
def rewire():
    m = _public(common())
    r = common()["_r"]
    enc, dec, j0 = A.blocks("enc"), A.blocks("dec"), A.blocks("judge0")
    if not (np.array_equal(j0["0.weight"], enc["0.weight"])
            and np.array_equal(j0["0.bias"], enc["0.bias"])):
        raise ValueError("the judge's body is not the encoder's")
    x, y = test_digits(REAL + 1)
    code = A.encode(enc, x[DIGIT:DIGIT + 1])[0]
    back = A.decode(dec, code[None])[0]
    made_code = np.asarray(r["fixed"][CODE_SHOWN], float)
    made = np.asarray(r["made0"][CODE_SHOWN], float)
    if not np.allclose(made, A.decode(dec, made_code[None].astype(np.float32))[0]):
        raise ValueError("made0 is not the decoder's own output")
    lab, sure = A.read(np.stack([x[DIGIT], back, made]))
    said = _sig(A.mlp(j0, np.stack([x[REAL], made]).astype(np.float32))[0][:, 0])
    eps = 1e-12
    loss = float(-(x[DIGIT] * np.log(np.maximum(back, eps))
                   + (1 - x[DIGIT]) * np.log(np.maximum(1 - back, eps))).mean())
    c = m["counts"]
    m.update(
        digit=square(x[DIGIT])[0], digit_label=int(y[DIGIT]), code=code.tolist(),
        code_text=[val(v, 1) for v in code], rebuilt=square(back)[0],
        rebuilt_read=int(lab[1]), digit_loss=loss, digit_loss_text=val(loss),
        pixels=784, made_code=made_code.tolist(), made=square(made)[0],
        made_read=int(lab[2]), made_sure=float(sure[2]), real=square(x[REAL])[0],
        real_label=int(y[REAL]), said_real=float(said[0]), said_made=float(said[1]),
        said_text=[val(v) for v in said], patches=patches(),
        kept=c["enc1"], left_out=c["enc2"], kept_text=commas(c["enc1"]),
        left_out_text=commas(c["enc2"]),
        evidence_numbers="exact arithmetic: in x out + out, each layer",
        limitations=[
            "one run, one seed; the digit is held-out digit %d (a %d), the made-up code the "
            "first of the 16 fixed codes, not picked for looks" % (DIGIT, int(y[DIGIT])),
            "the tiles drawn are corners of the real weight tensors (magnitudes), not all of "
            "them",
            "the judge's new head starts random (on the real %d and the made digit it says "
            "%s and %s, which means nothing yet); the GAN game then trains the judge and the "
            "generator on (animations 266-268)" % (int(y[REAL]), val(said[0]), val(said[1])),
            "the encoder's last layer (256 -> 16, %s numbers) is left out of the judge"
            % commas(c["enc2"])])
    panels = (
        Panel("Numbers in each block", "exact: in x out + out, per layer", "bars",
              dict(values=np.array([c["encoder"], c["decoder"], c["judge"], c["head"]], float),
                   labels=["encoder", "decoder = generator", "judge", "judge's new head"])),
        Panel("Squeezed and rebuilt, and made up", "held-out digit %d, its rebuild, and the "
              "decoder fed a made-up code" % DIGIT, "images",
              dict(values=np.stack([m["digit"], m["rebuilt"], m["made"]]))),
        Panel("Codes", "the digit's 16 numbers, and a made-up 16", "matrix",
              dict(values=np.stack([code, made_code]))),
        Panel("The judge's numbers", "kept from the encoder against new", "bars",
              dict(values=np.array([c["enc1"], c["head"]], float),
                   labels=["encoder's first layer, kept", "new head"])),
    )
    return Study("Same blocks, wired again",
                 "The generator is the decoder, all %s numbers kept; the judge is the "
                 "encoder's first layer (%s kept) and a new head of %d"
                 % (m["counts_text"]["decoder"], m["kept_text"], c["head"]), panels, m)


# ====================================================================== 266
def blurry():
    m = _public(common())
    r = common()["_r"]
    x, _y = test_digits(SHOWN)
    pics = dict(real=square(x), rebuilt=square(r["examples_rebuilt"][:SHOWN]),
                before=square(r["examples_before"][:SHOWN]),
                after=square(r["examples_after"][:SHOWN]))
    scores = pixel_scores()
    m.update(
        shown=pics, shown_grey={k: [float(grey_mask(p).mean()) for p in v]
                                for k, v in pics.items()},
        guesses=[0.1, 0.5, 0.9], scores=scores, scores_text=[val(s) for s in scores],
        limitations=[
            "one run each, one seed; grey = the share of pixels between 0.2 and 0.8, over "
            "1,000 held-out digits (real, rebuilt) and 1,000 made-up codes (the same codes "
            "before and after the GAN game)",
            "certainty: the course's MNIST classifier's top probability, averaged",
            "the pictures shown are the first three of each, not picked for looks; some "
            "made ones are speckled or malformed",
            "why rebuilds are grey is exact arithmetic on one pixel (binary cross-entropy is "
            "lowest at the share of digits with ink there), not a measurement"])
    k = m["kinds"]
    panels = (
        Panel("The pictures", "real, rebuilt, made before and after the game (first 3)",
              "images", dict(values=np.concatenate([pics[n] for n in k]))),
        Panel("Grey pixels", "share between 0.2 and 0.8, 1,000 of each", "bars",
              dict(values=np.array([m["grey"][n] for n in k]), labels=list(k))),
        Panel("Classifier's certainty", "mean top probability, 1,000 of each", "bars",
              dict(values=np.array([m["sure"][n] for n in k]), labels=list(k))),
        Panel("One pixel's score", "ink in half the digits: the loss of painting 0.1, 0.5, "
              "0.9", "bars", dict(values=np.array(scores), labels=["0.1", "0.5", "0.9"])),
    )
    return Study("Blurry or sharp",
                 "Grey pixels: real %s, rebuilt %s; the decoder on made-up codes %s before "
                 "the GAN game and %s after" % (m["grey_text"]["real"], m["grey_text"]["rebuilt"],
                                               m["grey_text"]["before"],
                                               m["grey_text"]["after"]), panels, m)


# ====================================================================== 267
def rebuild():
    m = _public(common())
    r = common()["_r"]
    x, y = test_digits(PAIRS)
    back = rebuilt(x)
    lab, _ = A.read(back)
    lab_real, _ = A.read(x)
    after = m["made_counts"]["after"]
    made = np.asarray(r["made%d" % A.GAN_STEPS], float)
    made_read, _ = A.read(made)
    m.update(
        pairs=dict(real=square(x), rebuilt=square(back)), pair_labels=[int(v) for v in y],
        pair_read=[int(v) for v in lab], pair_read_real=[int(v) for v in lab_real],
        after_min=int(min(after)), after_max=int(max(after)),
        after_min_digit=int(np.argmin(after)), after_max_digit=int(np.argmax(after)),
        made=square(made), made_read=[int(v) for v in made_read],
        made_code=np.asarray(r["fixed"][0], float).tolist(), patches_gen=patches(which_dec="gen"),
        limitations=[
            "one run, one seed; read by the course's MNIST classifier, itself right on 94.9% "
            "of these 1,000 real digits, so the counts are what it reads, not ground truth",
            "the made digits are counted from 1,000 made-up codes after 6,000 steps of the "
            "GAN game; the 16 shown are the fixed codes' pictures, as they are",
            "the GAN has no input for a digit: to make a given digit would need a "
            "conditional GAN, not shown"])
    panels = (
        Panel("Given and rebuilt", "the first %d held-out digits and their rebuilds" % PAIRS,
              "images", dict(values=np.concatenate([square(x), square(back)]))),
        Panel("Read as the right digit", "of 1,000 held-out digits: real, rebuilt", "bars",
              dict(values=np.array([m["real_right_n"], m["rebuild_right_n"]], float),
                   labels=["real", "rebuilt"])),
        Panel("Made digits, by what they are read as", "1,000 made-up codes after the game",
              "bars", dict(values=np.array(after, float), labels=[str(d) for d in range(10)])),
        Panel("Made from the 16 fixed codes", "after 6,000 steps, as they are", "images",
              dict(values=square(made))),
    )
    return Study("Rebuild or invent",
                 "The autoencoder's rebuilds are read as their digit %s of the time (real "
                 "digits %s); the GAN makes every digit, %d to %d of 1,000, but none on request"
                 % (m["rebuild_right_text"], m["real_right_text"], m["after_min"],
                    m["after_max"]), panels, m)


# ====================================================================== 268
def chunked(v, n=CHUNK):
    v = np.asarray(v, float)
    return v[:len(v) // n * n].reshape(-1, n).mean(1)


def losses():
    m = _public(common())
    r = common()["_r"]
    ae = np.asarray(r["ae_loss"], float)
    lj, lg = chunked(r["lj"]), chunked(r["lg"])
    snaps = {s: square(np.asarray(r["made%d" % s], float)[:SNAP_SHOWN]) for s in A.SNAPS}
    m.update(
        ae_loss=ae, ae_steps_x=np.arange(len(ae), dtype=float),
        lj_chunks=lj, lg_chunks=lg, chunk_x=(np.arange(len(lj), dtype=float) + 1) * CHUNK,
        coin_judge=2 * math.log(2), coin_gen=math.log(2), chunk=CHUNK,
        ae_first_text=val(m["ae_first"]), ae_last_text=val(m["ae_last"]),
        lj_text=(val(m["lj_first"]), val(m["lj_last"])),
        lg_text=(val(m["lg_first"]), val(m["lg_last"])),
        snaps=list(A.SNAPS), snap_pics=snaps,
        limitations=[
            "one run each, one seed; the GAN's losses are shown as means of 100 steps (each "
            "step's loss is on one batch of 64)",
            "2 ln 2 = 1.39 and ln 2 = 0.69 are the losses when the judge says 0.5 to "
            "everything (exact); they mark 'cannot tell', not a goal",
            "the GAN's losses do not say how good the pictures are; the pictures are shown "
            "at steps 0, 500, 2,000 and 6,000 from the same 2 made-up codes (the first two "
            "of the 16 fixed)"])
    panels = (
        Panel("The autoencoder's loss", "binary cross-entropy per step, 3,000 steps", "curve",
              dict(x=m["ae_steps_x"], ys=ae[None], labels=["loss"])),
        Panel("The GAN's two losses", "judge and generator, means of 100 steps", "curve",
              dict(x=m["chunk_x"], ys=np.stack([lj, lg]), labels=["judge", "generator"])),
        Panel("First and last 100 steps", "judge, generator", "bars",
              dict(values=np.array([m["lj_first"], m["lj_last"], m["lg_first"],
                                    m["lg_last"]]),
                   labels=["judge first", "judge last", "generator first",
                           "generator last"])),
        Panel("Made from the same codes", "steps 0, 500, 2,000, 6,000", "images",
              dict(values=np.concatenate([snaps[s] for s in A.SNAPS]))),
    )
    return Study("One loss or two players",
                 "The autoencoder's loss falls from %s to %s; the judge's goes %s to %s and "
                 "the generator's %s to %s, pulling against each other"
                 % (m["ae_first_text"], m["ae_last_text"], *m["lj_text"], *m["lg_text"]),
                 panels, m)


STUDIES = {"ae-to-gan-rewire": rewire, "ae-gan-blurry-or-sharp": blurry,
           "ae-gan-rebuild-or-invent": rebuild, "ae-gan-one-loss-or-two": losses}

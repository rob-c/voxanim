"""Build the project website: one static page showing every animation.

    python3 website/build_site.py OUT                     # videos from the latest release
    python3 website/build_site.py OUT --media build/out   # videos from a local render
    python3 -m http.server -d OUT                    # preview

The page is generated from the preset registry (names, output files and
descriptions) and :data:`SECTIONS` (how the animations are grouped and titled),
so it cannot drift from the code: a preset missing from :data:`SECTIONS`, or a
section naming one that does not exist, stops the build.

``OUT`` receives ``index.html``, the stylesheet and script, and for each
animation ``media/<file>.mp4`` (the full render, for the player and the
download), ``media/<file>-480.mp4`` (a 480 px preview at full frame rate,
played while a clip is hovered or focused), ``media/<file>-thumb.mp4`` (what
the grid plays all the time: the first half of the animation at 320 px and
12 frames a second, holding its last frame for 2 s before it loops; under
100 KB) and a poster frame ``media/<file>.jpg``.  The MP4s are not
committed to git: they come from a GitHub release (``gh release download``)
or from a local ``build/out``; previews, thumbnails and posters that already
exist in ``OUT`` are kept, so a cached ``media/`` makes a rebuild cheap.
Download links point at the same release's assets.  Needs
``ffmpeg``/``ffprobe``, and the GitHub CLI unless ``--media`` is given.  The
Pages workflow (``.github/workflows/pages.yml``) runs exactly this.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from voxanim import __version__, presets  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "rob-c/voxanim")
STATIC = ("style.css", "app.js")
HERO = "conv2d"
POSTER_AT = 0.4          # poster frame, as a fraction of the animation's length
POSTER_QUALITY = 4       # ffmpeg -q:v for the JPEG posters (2 best .. 31 worst)
PREVIEW = dict(width=480, crf=28, preset="medium")   # full-rate preview, on hover
PREVIEW_SUFFIX = "-480"
THUMB = dict(width=320, fps=12, crf=30, half=True, hold=2.0)  # what the grid plays all the time
THUMB_SUFFIX = "-thumb"

# (anchor, heading, one-line introduction, [(preset name, title), ...]).
# Every registered preset appears exactly once; the build checks it.
SECTIONS = [
    ("convolutions", "Convolutions",
     "Filters sliding over a block of channels: the conv2d set, then conv1d.",
     [("conv2d", "Convolution, 3×3 kernel"),
      ("conv2d-same-padding", "Same padding"),
      ("conv2d-stride-2", "Stride 2"),
      ("conv2d-stride-2-same-padding", "Stride 2 with same padding"),
      ("conv2d-2-groups", "Grouped convolution"),
      ("depthwise", "Depthwise convolution"),
      ("depthwise-separable", "Depthwise separable convolution"),
      ("conv1d-full-width-scroll", "Conv1D over a sequence"),
      ("conv1d-row-sweep", "Conv1D, row by row")]),
    ("lessons", "Lessons",
     "Short animations that each change one setting of a convolution.",
     [("lesson-stride-1", "Stride 1"),
      ("lesson-stride-2", "Stride 2"),
      ("lesson-stride-3", "Stride 3"),
      ("lesson-dilation-2", "Dilation 2"),
      ("lesson-dilation-3", "Dilation 3"),
      ("lesson-transposed-conv", "Transposed convolution"),
      ("lesson-max-pool", "Max pooling"),
      ("lesson-avg-pool", "Average pooling"),
      ("lesson-strided-downsample", "Learned downsampling"),
      ("lesson-pointwise", "Pointwise (1×1) convolution"),
      ("lesson-conv3d", "Conv3D"),
      ("lesson-conv3d-channels", "Conv3D with channels"),
      ("lesson-conv1d-stride-2", "Conv1D, stride 2"),
      ("lesson-conv1d-dilation-2", "Conv1D, dilation 2"),
      ("lesson-edge-detection", "Edge detection with Sobel filters")]),
    ("maths", "The maths, step by step",
     "Derivatives, probabilities and activations, with the real numbers in them.",
     [("autodiff-graph", "Automatic differentiation"),
      ("gpt2-softmax-cross-entropy", "Softmax and cross-entropy"),
      ("activations", "Activation functions")]),
    ("mlp", "An MLP, step by step",
     "A small MNIST network learning from one image, layer by layer, forward and back.",
     [("mlp-1-forward-layer1", "Forward: layer 1"),
      ("mlp-2-forward-layer2", "Forward: layer 2"),
      ("mlp-3-forward-output", "Forward: output and loss"),
      ("mlp-4-forward-pass", "The whole forward pass"),
      ("mlp-5-backward-output", "Backward: output layer"),
      ("mlp-6-backward-layer2", "Backward: layer 2"),
      ("mlp-7-backward-layer1", "Backward: layer 1"),
      ("mlp-8-backward-pass", "The whole backward pass"),
      ("mlp-9-training-step", "One training step")]),
    ("training", "How networks learn",
     "Forward pass, loss, gradients and update, for other kinds of network.",
     [("gradient-descent", "Gradient descent"),
      ("cnn-training-step", "A CNN training step"),
      ("rnn-training-step", "An RNN training step"),
      ("lstm-training-step", "An LSTM training step"),
      ("attention-training-step", "A self-attention training step"),
      ("rnn-batch-training", "Training an RNN in batches"),
      ("gated-memory", "Memory across a gap: RNN vs LSTM")]),
    ("optimisation", "Optimisation",
     "How gradient descent finds its way, and what the optimisers add.",
     [("loss-landscape-descent", "A loss landscape"),
      ("drunken-walk", "SGD: the drunken walk"),
      ("optimisers", "Optimisers on a narrow valley"),
      ("adamw-vs-l2", "AdamW is not Adam + L2"),
      ("learning-rate-schedule", "Learning-rate schedules")]),
    ("deep", "Training deep networks",
     "Why depth is hard, and the fixes.",
     [("overfitting", "Overfitting, and four cures"),
      ("initialisation", "Initialisation"),
      ("skip-connections", "Skip connections"),
      ("normalisation", "Batch norm and layer norm")]),
    ("capacity", "Capacity and bottlenecks",
     "How much a layer can carry, and what happens when it cannot.",
     [("bottleneck-sweep", "How narrow can the code be?"),
      ("information-bottleneck", "A layer that must forget"),
      ("rank-of-a-layer", "The rank of a layer"),
      ("width-vs-depth", "Width or depth?")]),
    ("classifiers", "Predictions",
     "Trained models deciding what they are looking at.",
     [("mlp-classifier", "A dense network classifies digits"),
      ("dnn-eval", "Evaluating a trained network"),
      ("strided-cnn-classifier", "A strided CNN classifier"),
      ("decision-tree", "Growing a decision tree")]),
    ("gp", "Gaussian processes",
     "Priors over functions, conditioned on data.",
     [("gp-prior", "A Gaussian process prior"),
      ("gp-posterior", "Conditioning on data"),
      ("gp-kernels", "Kernels are assumptions"),
      ("gp-hyperparameters", "Learning the hyperparameters"),
      ("gp-bayesian-optimisation", "Bayesian optimisation")]),
    ("latent", "Autoencoders",
     "Networks that squeeze a digit into a few numbers and rebuild it.",
     [("autoencoder-dense-3d", "Dense autoencoder, 3-D codes"),
      ("autoencoder-dense-2d", "Dense autoencoder, 2-D codes"),
      ("autoencoder-conv-3d", "Convolutional autoencoder, 3-D codes"),
      ("autoencoder-conv-2d", "Convolutional autoencoder, 2-D codes"),
      ("vae-dense-3d", "Variational autoencoder, 3-D codes"),
      ("vae-dense-2d", "Variational autoencoder, 2-D codes"),
      ("autoencoder-training", "An autoencoder learns its latent space"),
      ("diffusion", "Diffusion: noise and back"),
      ("pca-and-kmeans", "PCA and k-means"),
      ("unet-dense", "U-Net denoising"),
      ("dec-dense", "Deep embedded clustering"),
      ("idec-dense", "Improved deep embedded clustering")]),
    ("transformers", "Transformers",
     "Attention between two sequences, then GPT-2 small reading a real prompt, from one "
     "attention head to all 12 layers.",
     [("seq2seq-attention", "Attention between two sequences"),
      ("gpt2-tokenizer", "Byte-pair encoding"),
      ("gpt2-embeddings", "What embeddings know"),
      ("gpt2-attention", "Inside GPT-2"),
      ("gpt2-transformer-block", "Through the transformer"),
      ("gpt2-sampling", "Sampling the next token")]),
    ("behaviour", "What language models know, and don't",
     "Hallucination, conditioning and calibration, measured on GPT-2.",
     [("gpt2-hallucination", "Why a model makes things up"),
      ("gpt2-conditioning", "The prompt steers the distribution"),
      ("gpt2-calibration", "Confidence is not correctness")]),
    ("modern", "Modern sequence models",
     "State spaces, selection, experts and rotary positions.",
     [("ssm-recurrence", "State space models: one recurrence, two views"),
      ("mamba-selective-scan", "Mamba: selection"),
      ("context-cost", "The cost of context: attention, RNN, SSM"),
      ("mixture-of-experts", "Mixture of experts"),
      ("gpt2-rotary-embeddings", "Rotary position embeddings")]),
    ("rl", "Reinforcement learning",
     "An agent learning from reward.",
     [("q-learning", "Q-learning in a gridworld")]),
]


class SiteError(RuntimeError):
    """The site cannot be built as asked (a missing video, an unlisted preset)."""


def check_sections(sections=SECTIONS, registry=None):
    """Raise :class:`SiteError` unless every preset in ``registry`` (default:
    all registered) appears in ``sections`` exactly once, and nothing else."""
    registry = presets.REGISTRY if registry is None else registry
    listed = [name for *_, items in sections for name, _ in items]
    dupes = sorted({n for n in listed if listed.count(n) > 1})
    missing = sorted(set(registry) - set(listed))
    unknown = sorted(set(listed) - set(registry))
    if dupes or missing or unknown:
        raise SiteError("website SECTIONS out of date: duplicated %s, missing %s, unknown %s"
                        % (dupes, missing, unknown))


def prose(s):
    """A preset description as display text: ``3x3`` -> 3×3, ``->`` -> →."""
    s = re.sub(r"(?<=\d)x(?=\d)", "×", s)          # 3x3
    s = re.sub(r" x (?=\d)", " × ", s)              # 10 steps x 4 channels
    return s.replace("->", "→")


def describe(p):
    """``(first line, whole paragraph)`` of a preset's description."""
    about = " ".join(p.about.split())
    lines = [ln.strip() for ln in p.about.strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    # a first line wrapped mid-sentence: use the whole first sentence instead
    if first and not first.endswith((".", ":")):
        first = about.split(". ")[0].rstrip(".") + "."
    return prose(first), prose(about)


def release_tag(tag=None):
    """``tag``, or the tag of the repository's latest release (via ``gh``)."""
    if tag:
        return tag
    out = subprocess.run(["gh", "release", "view", "-R", REPO, "--json", "tagName",
                          "-q", ".tagName"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch_media(dest, media_dir, tag, files):
    """Put ``<file>.mp4`` for each of ``files`` into ``dest``: copied from
    ``media_dir``, or, when it is None, downloaded from release ``tag``."""
    os.makedirs(dest, exist_ok=True)
    if media_dir is None:
        subprocess.run(["gh", "release", "download", tag, "-R", REPO, "--pattern", "*.mp4",
                        "--dir", dest, "--skip-existing"], check=True)
    else:
        for f in files:
            src = os.path.join(media_dir, f + ".mp4")
            if os.path.exists(src) and not os.path.exists(os.path.join(dest, f + ".mp4")):
                shutil.copyfile(src, os.path.join(dest, f + ".mp4"))
    absent = [f for f in files if not os.path.exists(os.path.join(dest, f + ".mp4"))]
    if absent:
        raise SiteError("no video for: %s" % ", ".join(absent))


def duration(mp4):
    """Length of a video in seconds (ffprobe)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", mp4], check=True, capture_output=True, text=True)
    return float(out.stdout)


def poster(mp4, jpg, seconds):
    """Write the frame at ``seconds`` of ``mp4`` to ``jpg`` (kept if it exists)."""
    if os.path.exists(jpg):
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % seconds, "-i", mp4,
                    "-frames:v", "1", "-q:v", str(POSTER_QUALITY), jpg], check=True)


def _encode(mp4, out, filters, crf, before_input=()):
    """``mp4`` through ffmpeg ``filters`` to H.264 ``out``: silent, yuv420p,
    moov atom first so it starts at once.  Kept if ``out`` exists."""
    if os.path.exists(out):
        return
    tmp = out + ".part.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *before_input, "-i", mp4, "-vf", filters,
                    "-c:v", "libx264", "-preset", PREVIEW["preset"], "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, out)


def preview(mp4, out):
    """The full-rate preview of ``mp4``: :data:`PREVIEW` pixels wide."""
    _encode(mp4, out, "scale=%d:-2" % PREVIEW["width"], PREVIEW["crf"])


def thumb(mp4, out, seconds):
    """The grid's thumbnail of ``mp4`` (``seconds`` long): its first half at
    :data:`THUMB` frames a second, the last frame held ``hold`` seconds so the
    native loop pauses before it starts again."""
    t = THUMB
    cut = ("-t", "%.3f" % (seconds / 2)) if t["half"] else ()
    _encode(mp4, out, "fps=%d,scale=%d:-2,tpad=stop_mode=clone:stop_duration=%.1f"
            % (t["fps"], t["width"], t["hold"]), t["crf"], before_input=cut)


OPEN = ('<span class="open" aria-hidden="true"><svg viewBox="0 0 12 12" width="10" height="10">'
        '<path d="M2 1.5v9l8-4.5z" fill="currentColor"/></svg>Watch in full quality</span>')


def clip_data(name, title, p, secs, tag):
    """Everything the player dialog shows for one animation (its ``data-clip``)."""
    _first, about = describe(p)
    return {
        "name": name, "title": title, "about": about, "seconds": round(secs),
        "mp4": "media/%s.mp4" % p.media,
        "preview": "media/%s%s.mp4" % (p.media, PREVIEW_SUFFIX),
        "thumb": "media/%s%s.mp4" % (p.media, THUMB_SUFFIX),
        "poster": "media/%s.jpg" % p.media,
        "gif": "https://github.com/%s/releases/download/%s/%s.gif" % (REPO, tag, p.media),
        "manifest": "https://github.com/%s/releases/download/%s/%s.json" % (REPO, tag, p.media),
    }


def clip_html(name, title, p, secs, tag):
    """One animation in the grid: its frame with the "watch in full quality"
    cue, its title and summary.  The poster is a lazy <img>; app.js adds the
    <video> on top: the thumbnail while the clip is near the screen, the
    full-rate preview while it is hovered or focused."""
    first, _about = describe(p)
    data = clip_data(name, title, p, secs, tag)
    esc = html.escape
    return (
        '<li class="clip" id="%(name)s">\n'
        '  <a href="#%(name)s" data-clip=\'%(data)s\'>\n'
        '    <span class="frame" data-thumb="%(thumb)s" data-preview="%(preview)s">'
        '<img src="%(poster)s" width="960" height="540" loading="lazy" decoding="async" alt="">'
        '%(open)s</span>\n'
        '    <span class="clip-title">%(title)s</span>\n'
        '  </a>\n'
        '  <p>%(first)s</p>\n'
        '</li>' % dict(name=esc(name), data=esc(json.dumps(data, ensure_ascii=False), quote=True),
                       preview=esc(data["preview"]), thumb=esc(data["thumb"]),
                       poster=esc(data["poster"]), title=esc(title), first=esc(first), open=OPEN))


def page_html(durations, tag, sections=SECTIONS):
    """The whole page.  ``durations`` maps preset names to seconds."""
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as fh:
        template = fh.read()
    index, body = [], []
    for anchor, heading, intro, items in sections:
        index.append('<li><a href="#%s">%s</a></li>' % (anchor, html.escape(heading)))
        clips = "\n".join(clip_html(n, t, presets.REGISTRY[n], durations[n], tag)
                          for n, t in items)
        body.append(
            '<section class="group" id="%s" aria-labelledby="%s-h">\n'
            '<header><h2 id="%s-h">%s</h2><p>%s</p></header>\n'
            '<ol class="clips">\n%s\n</ol>\n</section>'
            % (anchor, anchor, anchor, html.escape(heading), html.escape(intro), clips))
    hero = presets.REGISTRY[HERO]
    hero_title = next(t for *_, items in sections for n, t in items if n == HERO)
    hero_data = clip_data(HERO, hero_title, hero, durations[HERO], tag)
    fill = {
        "count": str(len(presets.REGISTRY)), "version": __version__, "tag": html.escape(tag),
        "repo": REPO, "hero": hero.media, "hero_name": HERO, "index": "\n".join(index),
        "sections": "\n\n".join(body), "open": OPEN,
        "hero_clip": html.escape(json.dumps(hero_data, ensure_ascii=False), quote=True),
    }
    return re.sub(r"\{\{(\w+)\}\}", lambda m: fill[m.group(1)], template)


def build(out, media_dir=None, tag=None):
    """Write the site into ``out`` (see the module docstring)."""
    check_sections()
    tag = release_tag(tag)
    files = [p.media for p in presets.REGISTRY.values()]
    media = os.path.join(out, "media")
    fetch_media(media, media_dir, tag, files)
    durations = {}
    for name, p in presets.REGISTRY.items():
        mp4 = os.path.join(media, p.media + ".mp4")
        durations[name] = duration(mp4)
        poster(mp4, os.path.join(media, p.media + ".jpg"), POSTER_AT * durations[name])
        preview(mp4, os.path.join(media, p.media + PREVIEW_SUFFIX + ".mp4"))
        thumb(mp4, os.path.join(media, p.media + THUMB_SUFFIX + ".mp4"), durations[name])
    for f in STATIC:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(out, f))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page_html(durations, tag))
    # GitHub Pages: serve the files as they are, no Jekyll
    open(os.path.join(out, ".nojekyll"), "w").close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="output directory (created)")
    ap.add_argument("--media", help="take the MP4s from this directory (e.g. build/out) "
                                    "instead of downloading the release's")
    ap.add_argument("--tag", help="release the download links (and videos) come from "
                                  "(default: the latest)")
    args = ap.parse_args(argv)
    try:
        build(args.out, args.media, args.tag)
    except (SiteError, subprocess.CalledProcessError) as e:
        sys.exit("website: %s" % e)
    print("wrote %s" % os.path.join(args.out, "index.html"))


if __name__ == "__main__":
    main()

"""Build the project website: one static page showing every animation.

    python3 website/build_site.py OUT                     # media from the latest release
    python3 website/build_site.py OUT --media build/out   # media from a local render
    python3 -m http.server -d OUT                    # preview

The page is generated from the preset registry (names, output files and
descriptions) and :data:`SECTIONS` (how the animations are grouped and titled),
so it cannot drift from the code: a preset missing from :data:`SECTIONS`, or a
section naming one that does not exist, stops the build.

``OUT`` receives ``index.html``, the stylesheet and script, and for each animation
``media/<file>.jpg`` (its poster), ``media/<file>-thumb.mp4`` (what its card plays: the
first half of the animation, small and silent), the same two in dark mode
(``media/<file>-dark.jpg``, ``media/<file>-dark-thumb.mp4``; a card shows the pair that
matches the page's theme) and its render manifests (``media/<file>.json``,
``media/<file>-dark.json``).  Those are made when the animation is rendered
(:mod:`voxanim.previews`) and come here in one file of the release,
``voxanim-site.tar.gz`` (``tools/release_assets.py``): the build downloads that archive
and nothing else, and runs no ffmpeg.  The full videos are not on the site at all: the
player streams them from the same version's release of the theme's own repository
(``MEDIA_REPOS``: ``<repo>-light`` has ``<media>.mp4``, ``<repo>-dark`` has
``<media>-dark.mp4``), and the download links point there too.  An animation is offered
in dark mode when the archive has its dark manifest.

``--media DIR`` builds from a local folder instead, for a preview without a release: a
render folder (``build/out``) or one filled by ``tools/release_assets.py``.  The player
then plays DIR's own videos (linked into ``OUT/media``), and animations DIR does not
have are left off the page.  Needs the GitHub CLI unless ``--media`` is given.  The Pages
workflow (``.github/workflows/pages.yml``) runs exactly this.
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
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from voxanim import __version__, presets  # noqa: E402
from voxanim.previews import CLIP_SUFFIX, POSTER_SUFFIX  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "rob-c/voxanim")
STATIC = ("style.css", "app.js", "logo-mark.svg", "logo.svg", "logo-dark.svg")
HERO = "conv2d"
SITE_ARCHIVE = "voxanim-site.tar.gz"           # as tools/release_assets.py names them
GIF_ARCHIVE = "voxanim-gifs-%s.tar"
# The full videos and the GIF archives are not in this repository's release: each theme
# has a repository of its own, which holds nothing but a release per version (2.0.0: 598
# videos and 2.2 GB of GIFs would have sat beside the source's own release).
MEDIA_REPOS = {"light": REPO + "-light", "dark": REPO + "-dark"}
DARK = "-dark"                                 # voxanim.theme.suffix() of a dark render
# What the site archive may put on the site: flat names, these three kinds of file.
ARCHIVE_MEMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(jpg|mp4|json)")

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
      ("gpt2-softmax-cross-entropy", "Softmax and cross-entropy")]),
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
    ("generalisation", "Generalisation",
     "What makes a network work on data it has not seen.",
     [("double-descent", "Double descent"),
      ("memorising-noise", "Networks memorise anything"),
      ("data-vs-model-size", "Data against model size"),
      ("augmentation", "Data augmentation"),
      ("sharp-vs-flat-minima", "Sharp and flat minima")]),
    ("regularisation", "Regularisation",
     "L1, L2 and the other penalties that stop a network memorising, measured.",
     [("l1-vs-l2", "L1 against L2"),
      ("l1-l2-weights", "L1 and L2 on a network's weights"),
      ("regularisers", "Dropout, noise, max-norm, label smoothing"),
      ("regularisation-race", "Seven regularisers, one network"),
      ("regularised-autoencoder", "A regularised autoencoder")]),
    ("activation", "Activation functions",
     "ReLU, GELU, SiLU and the rest, training the same classifier.",
     [("activations", "Activation functions"),
      ("activation-race", "Seven activations, one network"),
      ("activation-gradients", "Gradients through each activation"),
      ("activation-sparsity", "What each activation leaves on"),
      ("activation-smoothness", "Kinks and curves"),
      ("activation-depth", "Activations and depth")]),
    ("capacity", "Capacity and bottlenecks",
     "How much a layer can carry, and what happens when it cannot.",
     [("bottleneck-sweep", "How narrow can the code be?"),
      ("information-bottleneck", "A layer that must forget"),
      ("rank-of-a-layer", "The rank of a layer"),
      ("width-vs-depth", "Width or depth?")]),
    ("pruning", "Trimming: pruning and distillation",
     "Making a trained network smaller without losing what it learned.",
     [("magnitude-pruning", "Pruning by magnitude"),
      ("lottery-ticket", "The lottery ticket"),
      ("structured-pruning", "Removing whole neurons and filters"),
      ("distillation", "Distillation")]),
    ("finetune", "Fine-tuning and transfer",
     "Starting from a trained network instead of from scratch.",
     [("transfer-learning", "Transfer learning"),
      ("catastrophic-forgetting", "Catastrophic forgetting"),
      ("learning-rate-finder-and-restarts", "Finding the learning rate, and restarting"),
      ("gpt2-lora", "LoRA: two thin matrices")]),
    ("efficiency", "Quantization and precision",
     "Fewer bits per number, and what it costs.",
     [("quantization", "Quantization"),
      ("quantization-aware-training", "Quantization-aware training"),
      ("mixed-precision", "float32, float16, bfloat16")]),
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
    ("embeddings", "Embeddings",
     "Tokens, words, sentences and image patches as vectors, and the ways a transformer is "
     "told where each one sits.",
     [("embedding-lookup", "An embedding is a row"),
      ("word2vec-training", "Training word2vec"),
      ("sentence-embeddings", "Sentence embeddings"),
      ("patch-embeddings", "Patch embeddings"),
      ("gpt2-position-blindness", "Attention cannot see order"),
      ("sinusoidal-positions", "Sinusoidal positions"),
      ("relative-position-bias", "Relative position bias"),
      ("alibi", "ALiBi"),
      ("rope-context-extension", "Stretching RoPE"),
      ("length-extrapolation", "Train short, test long")]),
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

# The "next 100" (voxanim.next_catalog): one section per group of the catalogue, each
# animation under the title it shows on screen. Their headings and introductions are here;
# their names, titles and descriptions come from the catalogue.
from voxanim.next_catalog import ITEMS as NEXT_ITEMS  # noqa: E402

NEXT_SECTIONS = {
    "A": ("attention-wiring", "Attention: how it is wired",
          "Who may read whom: masks, heads, caches and the ways attention is made cheaper."),
    "B": ("transformer-block", "The transformer block",
          "The design choices inside a block: where to normalise, what the MLP is, how wide and deep."),  # noqa: E501
    "C": ("training-language-models", "Training language models",
          "What a language model is trained on, how its loss is read, and what makes training work."),  # noqa: E501
    "D": ("decoding", "Decoding and inference",
          "Turning next-token chances into text, and serving it quickly."),
    "E": ("fine-tuning", "Fine-tuning and alignment",
          "Changing a trained model: instructions, preferences, small add-ons and weight arithmetic."),  # noqa: E501
    "F": ("interpretability", "Interpretability",
          "Looking inside a trained model: features, probes, patches and the heads that matter."),
    "G": ("vision", "Vision",
          "What image models see, attend to and get wrong."),
    "H": ("generative-models", "Generative models",
          "Models that make new samples: adversaries, flows, diffusion, codes and pixels in order."),  # noqa: E501
    "I": ("classical-ml", "Classical machine learning",
          "The ideas under everything else: fitting, validating, separating and measuring."),
    "J": ("optimisation-deeper", "Optimisation, deeper",
          "Curvature, noise and averaging: why training behaves as it does."),
    "K": ("retrieval", "Retrieval and contrastive learning",
          "Embeddings trained to bring matching things together, and searching among them."),
    "L": ("systems", "Systems and scale",
          "Spreading a model over devices, and what limits how fast it runs."),
    "M": ("reinforcement-learning-methods", "Reinforcement learning methods",
          "Learning from reward alone: bandits, policy gradients, values and search."),
    "N": ("memory-and-dynamics", "Memory, dynamics and curiosities",
          "Networks as memories, as rules that grow shapes, and as flows."),
    "O": ("generative-families", "Generative families compared",
          "An autoencoder, a VAE, a GAN and DEC trained on the same digits: objectives, codes, samples."),  # noqa: E501
    "P": ("label-embeddings", "Label embeddings",
          "A class label as a learned vector: in a classifier, in an autoencoder, and at generation."),  # noqa: E501
    "Q": ("randomness", "Randomness and entropy",
          "Where random numbers come from, what a seed decides, and how unpredictability "
          "is measured."),
    "R": ("probability", "Probability: frequentist and Bayesian",
          "Two readings of the same measurement, and the distributions every count and average follows."),  # noqa: E501
    "S": ("fitting", "Minimisation and chi-square fitting",
          "Fitting a model to data points with errors, and how a minimiser finds the best fit."),
    "T": ("likelihood", "Maximum likelihood and errors",
          "Where a best-fit value and its error come from."),
    "U": ("systematics", "Correlated errors and systematic uncertainties",
          "What changes when errors are shared between parameters or measurements."),
    "V": ("selection", "Data selection and significance",
          "Choosing events, and what the choice does to signal, background and the shape of the data."),  # noqa: E501
    "W": ("tests", "Hypothesis tests",
          "Deciding between two hypotheses, and the two ways of being wrong."),
    "X": ("discovery", "Discoveries, exclusion and upper limits",
          "Claiming a signal, or saying how large a signal could be hiding."),
    "Y": ("analysis-ml", "Machine learning in a physics analysis",
          "Networks and trees used the way an analysis uses them: to estimate a quantity or to select events."),  # noqa: E501
    "Z": ("anomaly-and-background", "Anomalies, collapse and smooth backgrounds",
          "An autoencoder as a detector of the unexpected, a GAN that forgets, and a Gaussian process as a background model."),  # noqa: E501
    # the alphabet is used up: later groups take lower-case letters
    "a": ("trees", "Decision trees, forests and boosting",
          "From one split to bagging, forests, boosting and the workings of XGBoost."),
    "b": ("information", "Entropy, impurity and loss",
          "One measure of uncertainty, in bits or nats: in a tree's leaves, in a classifier's "
          "loss, and between two distributions."),
}
# Group c, the bridges between topics, is not a section: each bridge opens the section it
# leads into (see _regroup).
BRIDGES = {item["preset"]: True for item in NEXT_ITEMS if item["section"].startswith("c. ")}
# Where each group sits on the page: the catalogue's order, except that the two groups added
# after the hundred follow the generative models they build on, not the end of the page.
NEXT_ORDER = "ABCDEFGHOPIQRSTUVWXYZJKLMNab"
assert sorted(NEXT_ORDER) == sorted(NEXT_SECTIONS)
for _letter in NEXT_ORDER:
    _anchor, _heading, _intro = NEXT_SECTIONS[_letter]
    SECTIONS.append((_anchor, _heading, _intro,
                     [(item["preset"], item["title"]) for item in NEXT_ITEMS
                      if item["section"].startswith(_letter + ". ")]))


class SiteError(RuntimeError):
    """The site cannot be built as asked (a missing video, an unlisted preset)."""


def check_sections(sections=None, registry=None):
    """Raise :class:`SiteError` unless every preset in ``registry`` (default:
    all registered) appears in ``sections`` exactly once, and nothing else."""
    registry = presets.REGISTRY if registry is None else registry
    sections = SECTIONS if sections is None else sections
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


def release_tag():
    """The tag of the repository's latest release (via ``gh``)."""
    out = subprocess.run(["gh", "release", "view", "-R", REPO, "--json", "tagName",
                          "-q", ".tagName"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def release_url(tag, name, theme=None):
    """Where release ``tag`` serves its file ``name``: this repository's release, or with
    ``theme`` the same tag's release of that theme's media repository."""
    return "https://github.com/%s/releases/download/%s/%s" % (
        MEDIA_REPOS[theme] if theme else REPO, tag, name)


def unpack_site(archive, dest):
    """Unpack a ``voxanim-site.tar.gz`` into ``dest``; returns how many files.  Only plain
    files with flat, ordinary names are taken: what is unpacked is served to the public,
    and the archive is a download."""
    os.makedirs(dest, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not ARCHIVE_MEMBER.fullmatch(member.name):
                raise SiteError("%s holds %r, which the site archive should not"
                                % (os.path.basename(archive), member.name))
            with tf.extractfile(member) as src, \
                    open(os.path.join(dest, member.name), "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def fetch_site(dest, tag):
    """Download release ``tag``'s site archive and unpack it into ``dest``."""
    with tempfile.TemporaryDirectory() as tmp:
        got = subprocess.run(["gh", "release", "download", tag, "-R", REPO, "--pattern",
                              SITE_ARCHIVE, "--dir", tmp], capture_output=True, text=True)
        if got.returncode != 0:
            # a release made before 1.11 has a video per animation and no archive
            raise SiteError("release %s has no %s to build from (%s); publish a release "
                            "gathered by tools/release_assets.py"
                            % (tag, SITE_ARCHIVE, got.stderr.strip() or "gh failed"))
        return unpack_site(os.path.join(tmp, SITE_ARCHIVE), dest)


def _link(src, dst):
    """``dst`` as a link to ``src`` (a copy where links are not allowed): a local preview
    plays the folder's own videos without a second 1.2 GB of them."""
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copyfile(src, dst)


def local_media(dest, media_dir, files):
    """Fill ``dest`` from the local folder ``media_dir`` for each of ``files`` (media
    names): the posters, clips and manifests from its site archive if it has one (a folder
    filled by ``tools/release_assets.py``), else as loose files (``build/out``); and links
    to its videos, which a local preview plays in place of the release's."""
    os.makedirs(dest, exist_ok=True)
    # release_assets.py fills a folder per repository; a render folder has no such folders
    dirs = [media_dir] + [d for d in (os.path.join(media_dir, os.path.basename(r))
                                      for r in (REPO, *MEDIA_REPOS.values()))
                          if os.path.isdir(d)]
    for d in dirs:
        archive = os.path.join(d, SITE_ARCHIVE)
        if os.path.exists(archive):
            unpack_site(archive, dest)
    for f in files:
        for d in dirs:
            for tail in (POSTER_SUFFIX, CLIP_SUFFIX, ".json", DARK + POSTER_SUFFIX,
                         DARK + CLIP_SUFFIX, DARK + ".json"):
                src = os.path.join(d, f + tail)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(dest, f + tail))
            for tail in (".mp4", DARK + ".mp4"):
                src = os.path.join(d, f + tail)
                if os.path.exists(src):
                    _link(src, os.path.join(dest, f + tail))


def _seconds(manifest):
    """An animation's length from its render manifest, or ``None`` if the manifest is from
    before lengths were recorded (1.10 and earlier)."""
    with open(manifest, encoding="utf-8") as fh:
        m = json.load(fh)
    return m["frames"] / m["fps"] if m.get("frames") and m.get("fps") else None


def survey(media, local=False, registry=None):
    """What ``media`` (the site's media folder, filled) can show: ``{preset name:
    {"seconds": length, "dark": has a dark video, "dark_card": has a dark poster and
    clip}}`` for every animation with a poster, a clip and a manifest that gives its
    length.  The rest are left off the page: an
    animation added since the release has nothing there yet, and the site built on a push
    must not fail (or show it) until the release that carries it is published.  ``local``:
    the videos are played from ``media`` too, so they must be there."""
    registry = presets.REGISTRY if registry is None else registry
    found = {}
    for name, p in registry.items():
        def has(tail, media_name=p.media):
            return os.path.exists(os.path.join(media, media_name + tail))
        if not (has(POSTER_SUFFIX) and has(CLIP_SUFFIX) and has(".json")):
            continue
        if local and not has(".mp4"):
            continue
        seconds = _seconds(os.path.join(media, p.media + ".json"))
        if seconds is None and local:           # an old local render: ask the video
            from voxanim.previews import duration
            seconds = duration(os.path.join(media, p.media + ".mp4"))
        if seconds is None:
            continue
        found[name] = {"seconds": seconds,
                       "dark": has(DARK + ".json") and (not local or has(DARK + ".mp4")),
                       "dark_card": has(DARK + POSTER_SUFFIX) and has(DARK + CLIP_SUFFIX)}
    return found


OPEN = ('<span class="open" aria-hidden="true"><svg viewBox="0 0 12 12" width="10" height="10">'
        '<path d="M2 1.5v9l8-4.5z" fill="currentColor"/></svg>Watch in full quality</span>')


def clip_data(name, title, p, info, tag, local=False):
    """Everything the card and the player dialog show for one animation (its
    ``data-clip``).  ``video`` and ``manifest`` map each theme the animation has to its
    file: the videos on the release (or, for a ``local`` preview, in the site's own media
    folder), the manifests on the site.  ``poster`` and ``thumb`` map each theme the card
    has previews in to them; light is always there, and app.js falls back to it."""
    _first, about = describe(p)
    themes = [("light", "")] + ([("dark", DARK)] if info["dark"] else [])
    cards = [("light", "")] + ([("dark", DARK)] if info.get("dark_card") else [])

    def video(theme, tail):
        f = "%s%s.mp4" % (p.media, tail)
        return "media/" + f if local else release_url(tag, f, theme)

    return {
        "name": name, "title": title, "about": about, "seconds": round(info["seconds"]),
        "poster": {theme: "media/%s%s%s" % (p.media, tail, POSTER_SUFFIX)
                   for theme, tail in cards},
        "thumb": {theme: "media/%s%s%s" % (p.media, tail, CLIP_SUFFIX)
                  for theme, tail in cards},
        "video": {theme: video(theme, tail) for theme, tail in themes},
        "manifest": {theme: "media/%s%s.json" % (p.media, tail) for theme, tail in themes},
    }


def clip_html(name, title, p, info, tag, local=False):
    """One animation in the grid: its frame with the "watch in full quality"
    cue, its title and summary.  The poster is a lazy <img>; app.js adds the
    small clip as a <video> on top while the card is near the screen."""
    first, _about = describe(p)
    data = clip_data(name, title, p, info, tag, local)
    esc = html.escape
    return (
        '<li class="clip" id="%(name)s">\n'
        '  <a href="#%(name)s" data-clip=\'%(data)s\'>\n'
        '    <span class="frame" %(frame)s>%(picture)s%(open)s</span>\n'
        '    <span class="clip-title">%(title)s</span>\n'
        '  </a>\n'
        '  <p>%(first)s</p>\n'
        '</li>' % dict(name=esc(name), data=esc(json.dumps(data, ensure_ascii=False), quote=True),
                       frame=frame_attrs(data), title=esc(title), first=esc(first), open=OPEN,
                       picture=picture_html(data, 'loading="lazy" decoding="async" alt=""')))


def frame_attrs(data):
    """The attributes app.js reads on a card's frame: the clip of each theme it has."""
    out = 'data-thumb="%s"' % html.escape(data["thumb"]["light"])
    if "dark" in data["thumb"]:
        out += ' data-thumb-dark="%s"' % html.escape(data["thumb"]["dark"])
    return out


def picture_html(data, attrs):
    """A card's poster.  Where there is a dark one, the browser picks by the system's
    colour scheme before any script runs, so a dark page never downloads the light poster
    first; app.js takes over when the page's own "Dark mode" button has chosen."""
    light, dark = data["poster"]["light"], data["poster"].get("dark")
    img = '<img src="%s" width="640" height="360" %s>' % (html.escape(light), attrs)
    if not dark:
        return img
    return ('<picture data-dark="%s"><source media="(prefers-color-scheme: dark)" srcset="%s">'
            '%s</picture>' % (html.escape(dark), html.escape(dark), img))


def released(sections, found):
    """``sections`` with only the animations that have media (are in ``found``),
    and without the sections that are left empty."""
    kept = [(a, h, i, [(n, t) for n, t in items if n in found]) for a, h, i, items in sections]
    return [section for section in kept if section[3]]


# ------------------------------------------------------------------ the order of the page
# The page reads as one course, from a first look at probability to large language models:
# statistics, trees and classical methods, Gaussian processes, networks from first principles,
# convolutions, models in use, autoencoders and generative models, then sequences, attention,
# language models and what comes after the transformer. The sections above are written down in
# the order they were built; this is the order they are read in.

def _regroup(sections):
    """A few animations change section so each part can be read without the next: the decision
    tree joins the classical methods it belongs with, the recurrent networks get a section of
    their own ahead of attention, and the attention training step joins the transformers.
    Nothing is shown before what it rests on: the CNNs wait for convolutions, the position
    encodings for attention, and the bridges (catalogue group c) each open the section they
    lead into."""
    by = {anchor: [anchor, heading, intro, list(items)]
          for anchor, heading, intro, items in sections}
    titles = {item["preset"]: item["title"] for item in NEXT_ITEMS}

    def take(anchor, names):
        moved = [pair for pair in by[anchor][3] if pair[0] in names]
        assert len(moved) == len(names), (anchor, names)
        by[anchor][3] = [pair for pair in by[anchor][3] if pair[0] not in names]
        return moved

    def pull(*names):
        """``names`` taken out of whichever section holds them, in the order given; a bridge
        is in none yet and comes from the catalogue."""
        moved = []
        for name in names:
            held = [a for a, *_, items in by.values() if any(n == name for n, _ in items)]
            if held:
                moved += take(held[0], {name})
            else:
                assert BRIDGES.get(name), name
                moved.append((name, titles[name]))
        return moved

    def put(anchor, at, *names):
        """``names`` into section ``anchor`` at index ``at`` (None: at the end)."""
        moved = pull(*names)
        items = by[anchor][3]
        at = len(items) if at is None else at
        items[at:at] = moved

    by["classical-ml"][3][:0] = take("classifiers", {"decision-tree"})
    by["classical-ml"][1] = "Decision trees and classical machine learning"
    by["classical-ml"][2] = ("From an analysis cut to a model you can read, then the classical "
                             "ideas every later model is judged by.")
    put("classical-ml", 0, "from-a-cut-to-a-tree")
    put("classical-ml", 2, "every-loss-is-a-likelihood")
    by["recurrent"] = ["recurrent", "Recurrent networks",
                       "Networks that read a sequence one step at a time and carry a memory along.",
                       take("training", {"rnn-training-step", "lstm-training-step",
                                         "rnn-batch-training", "gated-memory"})]
    put("recurrent", 0, "markov-chain-to-rnn")
    by["training"][2] = "Forward pass, loss, gradients and update, one step at a time."
    by["transformers"][3][1:1] = take("training", {"attention-training-step"})

    # the arithmetic of impurity ahead of the first split that uses it; the weights' entropy
    # with the other measures of what a layer holds
    put("trees", 0, "entropy-and-gini-by-hand")
    put("capacity", 3, "weight-entropy")
    # the CNNs once convolutions are known
    put("vision", 0, "cnn-training-step", "strided-cnn-classifier")
    by["vision"][2] = "Training a CNN, then what image models see, attend to and get wrong."
    put("convolutions", 0, "dense-layer-learns-local")
    by["convolutions"][2] = ("Why a dense layer gives way to a filter, then the filter sliding "
                             "over a block of channels: the conv2d set, then conv1d.")
    # text becomes tokens, then rows; where each token sits waits for attention
    put("embeddings", 0, "gpt2-tokenizer")
    by["embeddings"][2] = ("Text cut into tokens, and tokens, words, sentences and image "
                           "patches as vectors.")
    by["positions"] = ["positions", "Where each token sits",
                       "Attention cannot see order: the ways a transformer is told where each "
                       "token sits, and how far that stretches.", []]
    put("positions", None, "gpt2-position-blindness", "sinusoidal-positions",
        "gpt2-rotary-embeddings", "relative-position-bias", "alibi", "rope-context-extension",
        "length-extrapolation")
    put("modern", 0, "context-cost")
    by["modern"][2] = "The cost of context, then state spaces, selection and experts."
    # one ladder for hardware: number formats, one device, its limits, then many devices
    put("efficiency", 0, "bits-in-a-weight")
    put("systems", 0, "cpu-and-gpu-training", "roofline")
    by["systems"][2] = ("One device and what limits it, then a model spread over many.")
    # learning from reward starts from the classifier it replaces
    put("rl", 0, "labels-against-rewards")
    by["rl"][2] = "From labels to rewards, then an agent learning from reward alone."
    put("latent", 0, "pca-and-kmeans")
    put("memory-and-dynamics", None, "hopfield-network")
    return by


PARTS = (
    ("Statistics for data analysis",
     "Probability, fits, errors, tests and limits: what a measurement says, and how sure it is.",
     ("probability", "randomness", "fitting", "likelihood", "systematics", "selection", "tests",
      "discovery")),
    ("Trees and classical machine learning",
     "The first models: ones you can read, and the ideas used to judge every model after them.",
     ("classical-ml", "trees", "information")),
    ("Gaussian processes",
     "A different kind of model: a distribution over functions, with its uncertainty built in.",
     ("gp",)),
    ("Neural networks from first principles",
     "One small network taken apart: the sums, the gradients, and what makes training work.",
     ("maths", "mlp", "training", "classifiers", "activation", "optimisation", "deep",
      "generalisation", "regularisation", "capacity")),
    ("Convolutions and vision",
     "Networks for images: a filter sliding over a picture, and what is built from it.",
     ("convolutions", "lessons", "vision")),
    ("Models in use",
     "Networks and trees doing a job, and the craft of training, shrinking and running them.",
     ("analysis-ml", "optimisation-deeper", "pruning", "finetune", "efficiency", "systems")),
    ("Other ways to learn",
     "Learning from reward instead of labels, and a few systems that learn in their own way.",
     ("rl", "reinforcement-learning-methods", "memory-and-dynamics")),
    ("Autoencoders and generative models",
     "From squeezing a picture into a few numbers to drawing new ones: AE, VAE, GAN and DEC.",
     ("latent", "generative-families", "label-embeddings", "generative-models",
      "anomaly-and-background")),
    ("Sequences and recurrent networks",
     "Data that comes in order: networks with a memory, and words turned into vectors.",
     ("recurrent", "embeddings")),
    ("Attention and transformers",
     "Every position looking at every other, and the block that GPT-2 stacks twelve times.",
     ("transformers", "positions", "attention-wiring", "transformer-block")),
    ("Large language models",
     "Training them, getting text out of them, steering them, and looking inside, with GPT-2.",
     ("training-language-models", "decoding", "behaviour", "fine-tuning", "interpretability",
      "retrieval")),
    ("After the transformer",
     "State-space models and other large sequence models built to go beyond it.",
     ("modern",)),
)

_grouped = _regroup(SECTIONS)
assert sorted(a for *_, anchors in PARTS for a in anchors) == sorted(_grouped), "a section is lost"
SECTIONS = [tuple(_grouped[a]) for *_, anchors in PARTS for a in anchors]
PART_OF = {a: k for k, (*_, anchors) in enumerate(PARTS) for a in anchors}


HERO_ALT = "A 3×3 convolution: eight filters sliding over a 7×7×8 input"


def page_html(found, tag, sections=None, local=False):
    """The whole page.  ``found`` is :func:`survey`'s answer: an animation that is not in
    it (not in the release yet) is left off the page.  ``local``: the player plays the
    videos in the site's own media folder, not the release's."""
    sections = released(SECTIONS if sections is None else sections, found)
    if not sections:
        raise SiteError("no animation has a poster, a preview clip and a manifest")
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as fh:
        template = fh.read()
    index, body, part = [], [], None
    for anchor, heading, intro, items in sections:
        if PART_OF.get(anchor, part) != part:       # the first section of a part: say so
            part = PART_OF[anchor]
            body.append('<div class="part" id="part-%d">\n<p class="part-n">Part %d of %d</p>\n'
                        '<p class="part-name">%s</p>\n<p class="part-intro">%s</p>\n</div>'
                        % (part + 1, part + 1, len(PARTS), html.escape(PARTS[part][0]),
                           html.escape(PARTS[part][1])))
        index.append('<li><a href="#%s">%s</a></li>' % (anchor, html.escape(heading)))
        clips = "\n".join(clip_html(n, t, presets.REGISTRY[n], found[n], tag, local)
                          for n, t in items)
        body.append(
            '<section class="group" id="%s" aria-labelledby="%s-h">\n'
            '<header><h2 id="%s-h">%s</h2><p>%s</p></header>\n'
            '<ol class="clips">\n%s\n</ol>\n</section>'
            % (anchor, anchor, anchor, html.escape(heading), html.escape(intro), clips))
    titles = {n: t for *_, items in sections for n, t in items}
    # a local preview of a few animations may not have the usual one to lead with
    lead = HERO if HERO in titles else next(iter(titles))
    hero = presets.REGISTRY[lead]
    hero_data = clip_data(lead, titles[lead], hero, found[lead], tag, local)
    fill = {
        "count": str(len(found)), "version": __version__, "tag": html.escape(tag),
        "parts": str(len(PARTS)),
        "repo": REPO, "hero": hero.media, "hero_name": lead, "index": "\n".join(index),
        "sections": "\n\n".join(body), "open": OPEN,
        "hero_frame": frame_attrs(hero_data),
        "hero_picture": picture_html(hero_data, 'fetchpriority="high" alt="%s"' % html.escape(
            HERO_ALT if lead == HERO else titles[lead])),
        "hero_clip": html.escape(json.dumps(hero_data, ensure_ascii=False), quote=True),
        "first": sections[0][0],
        "gifs_light": release_url(tag, GIF_ARCHIVE % "light", "light"),
        "gifs_dark": release_url(tag, GIF_ARCHIVE % "dark", "dark"),
        # a light-only release has no dark archive to link to
        "gifs_dark_hidden": "" if any(i["dark"] for i in found.values()) else " hidden",
    }
    return re.sub(r"\{\{(\w+)\}\}", lambda m: fill[m.group(1)], template)


def build(out, media_dir=None, tag=None):
    """Write the site into ``out`` (see the module docstring)."""
    check_sections()
    local = media_dir is not None
    # a local preview asks GitHub nothing: its release links name this version's tag
    tag = tag or ("v" + __version__ if local else release_tag())
    media = os.path.join(out, "media")
    if local:
        local_media(media, media_dir, [p.media for p in presets.REGISTRY.values()])
    else:
        fetch_site(media, tag)
    found = survey(media, local)
    absent = sorted(set(presets.REGISTRY) - set(found))
    if absent:
        print("website: %d animations have no media in %s and are left off the page: %s"
              % (len(absent), media_dir if local else "release " + tag, ", ".join(absent)))
    page = page_html(found, tag, local=local)
    for f in STATIC:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(out, f))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    # GitHub Pages: serve the files as they are, no Jekyll
    open(os.path.join(out, ".nojekyll"), "w").close()
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="output directory (created)")
    ap.add_argument("--media", help="build from this local folder (build/out, or one filled "
                                    "by tools/release_assets.py) instead of the release, "
                                    "and play its videos")
    ap.add_argument("--tag", help="release the media and the download links come from "
                                  "(default: the latest; with --media, this version's)")
    args = ap.parse_args(argv)
    try:
        found = build(args.out, args.media, args.tag)
    except (SiteError, subprocess.CalledProcessError, tarfile.TarError) as e:
        sys.exit("website: %s" % e)
    print("wrote %s (%d animations, %d with a dark video)"
          % (os.path.join(args.out, "index.html"), len(found),
             sum(i["dark"] for i in found.values())))


if __name__ == "__main__":
    main()

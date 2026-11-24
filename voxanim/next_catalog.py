"""The "next" animations (the hundred of TODO.md and the groups added after them), registered
lazily and kept out of release actions.

Each has a study (its measurements: :mod:`voxanim.next_attention` and its siblings) and a
:class:`~voxanim.mech.Mechanism` that draws it (``voxanim.mech_*``).
"""

from __future__ import annotations

import json
from functools import partial
from importlib import import_module
from pathlib import Path

from .presets import preset

GPT2 = {
    "kv-cache",
    "attention-sinks",
    "residual-stream",
    "perplexity-explained",
    "beam-search",
    "speculative-decoding",
    "repetition-and-penalties",
    "constrained-decoding",
    "contrastive-decoding",
    "knowledge-editing",
    "sparse-autoencoder-features",
    "linear-probes",
    "activation-patching",
    "induction-heads",
    "head-ablation",
    "retrieval-augmented-generation",
    "seeds-and-sampled-text",
}
MNIST = {
    "second-order-preconditioners",
    "evolution-strategies",
    "saliency-and-integrated-gradients",
    "feature-visualisation",
    "receptive-field",
    "vit-attention",
    "clip-contrastive",
    "adversarial-examples",
    "object-detection-basics",
    "semantic-segmentation",
    "batchnorm-in-cnns",
    "style-transfer",
    "classifier-free-guidance",
    "ddpm-vs-ddim",
    "vae-posterior-collapse",
    "vq-vae-codebook",
    "autoregressive-pixels",
    "tsne-and-umap",
    "roc-and-calibration",
    "edge-of-stability",
    "batch-size-and-noise",
    "weight-averaging",
    "hopfield-network",
    "neural-cellular-automata",
}
ITEMS = json.loads(Path(__file__).with_name("next_catalog.json").read_text())
for item in ITEMS:
    item["preset"] = ("gpt2-" if item["name"] in GPT2 else "") + item["name"]
# The revised learning studies use collected datasets. Analytic examples keep
# no-data status; a missing text download must not masquerade as an available run.
MNIST |= {item["name"] for item in ITEMS if 51 <= item["number"] <= 74}
MNIST |= {
    "grokking",
    "superposition",
    "newton-vs-gradient-descent",
    "vector-search-hnsw",
    "dimensionality-of-embeddings",
    "data-parallel-allreduce",
    "tensor-and-pipeline-parallel",
    "multi-armed-bandits",
    "policy-gradient",
    "actor-critic",
    "dqn-replay-buffer",
    "reward-shaping",
    "neural-ode",
    "activation-checkpointing",
}
MNIST |= {item["name"] for item in ITEMS if 100 < item["number"] <= 109}
MNIST |= {"seeds-and-training-outcomes"}
# machine learning in a physics analysis, and the anomaly and collapse animations
MNIST |= {"network-regression", "classifier-score-as-variable", "boosted-regression-trees",
          "overtraining-check", "autoencoder-anomaly-detection", "gan-mode-collapse",
          "anomaly-loss-and-latent", "anomaly-two-score-flow", "cpu-and-gpu-training"}
# trees and boosting, then the KL term and DEC's centroids, then entropy, impurity and loss
MNIST |= {item["name"] for item in ITEMS if 160 <= item["number"] <= 175}
# the bridges between topics: all on MNIST but the chain-to-RNN one, which reads real text
MNIST |= {item["name"] for item in ITEMS if 176 <= item["number"] <= 182} - {"markov-chain-to-rnn"}
# counting the weights: the DNN and CNN are checked against their trained arrays
MNIST |= {"counting-weights-dnn-and-cnn"}
# blocks and code: one MNIST classifier trained once (study_course) and shown block by block
MNIST |= {item["name"] for item in ITEMS if 185 <= item["number"] <= 214}
# blocks and code, a CNN: the same program with a CNN for its model (study_course_cnn)
MNIST |= {item["name"] for item in ITEMS if 215 <= item["number"] <= 223}
TEXT = {
    item["name"]
    for item in ITEMS
    if item["number"] <= 50
    and item["name"] not in MNIST | {"flash-attention-tiling", "batching-and-throughput"}
}
TEXT |= {"loss-spikes", "retrieval-augmented-generation"}
# the randomness group: real text for the chain and for measuring entropy
TEXT |= {"markov-chains", "measuring-entropy", "seeds-and-sampled-text"}
TEXT |= {"markov-chain-to-rnn"}
NO_DATA = {item["preset"] for item in ITEMS if item["name"] not in GPT2 | MNIST | TEXT}


def redesigned(name):
    """The replacement experiment for ``name``, if its first one was redesigned: a function
    returning the :class:`~voxanim.study.Study`, from ``voxanim.study_<section>.STUDIES``.

    The first experiments live in the ``next_*`` modules. Some could not show their concept
    (a draft model too weak for speculative decoding to pay, two passages to rank 144 heads) or
    came out null at their size; their replacements are kept per section, beside the drawing."""
    for module, first, last in DRAWN_BY:
        if first <= NUMBER[name] <= last:
            target = "voxanim.study_" + module[len("mech_"):]
            try:
                return import_module(target).STUDIES.get(name)
            except ModuleNotFoundError as missing:
                if missing.name != target:
                    raise
    return None


def build_study(module, name):
    replacement = redesigned(name)
    if replacement is not None:
        return replacement().validate()
    return import_module("voxanim." + module).make(name).validate()


# The modules that draw them (:mod:`voxanim.mech`), by catalogue number: each maps the
# names it draws to a :class:`~voxanim.mech.Mechanism`, built from that name's study.
DRAWN_BY = (
    ("mech_attention_a", 1, 6),
    ("mech_attention_b", 7, 12),
    ("mech_block", 13, 20),
    ("mech_training", 21, 28),
    ("mech_decoding", 29, 34),
    ("mech_finetune", 35, 42),
    ("mech_interp", 43, 50),
    ("mech_vision", 51, 58),
    ("mech_generative", 59, 66),
    ("mech_classical", 67, 74),
    ("mech_optim", 75, 80),
    ("mech_retrieval", 81, 85),
    ("mech_systems", 86, 90),
    ("mech_rl", 91, 96),
    ("mech_curios", 97, 100),
    ("mech_families", 101, 105),
    ("mech_labels", 106, 109),
    ("mech_random", 110, 114),
    ("mech_chance", 115, 117),
    ("mech_stats_prob", 118, 123),
    ("mech_stats_fit", 124, 127),
    ("mech_stats_likelihood", 128, 132),
    ("mech_stats_syst", 133, 136),
    ("mech_stats_select", 137, 140),
    ("mech_stats_tests", 141, 144),
    ("mech_stats_discovery", 145, 149),
    ("mech_phys_apps", 150, 153),
    ("mech_phys_adv", 154, 156),
    ("mech_anomaly", 157, 158),
    ("mech_hardware", 159, 159),
    ("mech_trees", 160, 164),
    ("mech_boost", 165, 169),
    ("mech_kl", 170, 171),
    ("mech_info", 172, 175),
    ("mech_bridge_kl", 176, 176),
    ("mech_bridge_loss", 177, 177),
    ("mech_bridge_cut", 178, 178),
    ("mech_bridge_local", 179, 179),
    ("mech_bridge_text", 180, 180),
    ("mech_bridge_reward", 181, 181),
    ("mech_bridge_bits", 182, 182),
    ("mech_count_nets", 183, 183),
    ("mech_count_attention", 184, 184),
    ("mech_blocks_a", 185, 187),
    ("mech_blocks_b", 188, 190),
    ("mech_blocks_c", 191, 193),
    ("mech_blocks_d", 194, 196),
    ("mech_blocks_e", 197, 199),
    ("mech_blocks_f", 200, 202),
    ("mech_blocks_g", 203, 205),
    ("mech_blocks_h", 206, 208),
    ("mech_blocks_i", 209, 211),
    ("mech_blocks_j", 212, 214),
    ("mech_blocks_k", 215, 217),
    ("mech_blocks_l", 218, 220),
    ("mech_blocks_m", 221, 223),
)
NUMBER = {item["name"]: item["number"] for item in ITEMS}


def drawing(name):
    """The class that draws ``name``. Only that animation's own module is imported."""
    for module, first, last in DRAWN_BY:
        if first <= NUMBER[name] <= last:
            return import_module("voxanim." + module).ANIMATIONS.get(name)
    return None


def build(module, name):
    """The animation ``name``: its study's measurements, drawn by its own class."""
    return drawing(name)(build_study(module, name))


for item in ITEMS:
    # the first line is what the list and the website show; the rest says what was measured
    preset(item["preset"], item["preset"], about=item["summary"] + "\n\n" + item["description"])(
        partial(build, item["module"], item["name"])
    )

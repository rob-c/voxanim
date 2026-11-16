"""Collected datasets and replayable evidence for the unpublished TODO studies.

Text is split into disjoint contiguous portions before sampling windows. MNIST
keeps its official test split and reserves its last 10,000 training images for
validation. Random corruption of an observed example is not a new data source.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cache import DEFAULT_CACHE, download, sha256_of

ROOT = Path(__file__).resolve().parents[1]
TEXT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TEXT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def source_hash(*names):
    h = hashlib.sha256()
    for name in sorted(set(names)):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / "voxanim" / name
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def _text_parts():
    path = Path(DEFAULT_CACHE) / "tinyshakespeare.txt"
    download(TEXT_URL, str(path), TEXT_SHA256)
    if sha256_of(path) != TEXT_SHA256:
        raise ValueError("Tiny Shakespeare checksum does not match the pinned dataset")
    content = path.read_text(encoding="utf-8")
    cuts = [
        0,
        content.find("\n", int(len(content) * 0.8)) + 1,
        content.find("\n", int(len(content) * 0.9)) + 1,
        len(content),
    ]
    return {
        key: (content[a:b], a, b)
        for key, a, b in zip(("train", "validation", "test"), cuts, cuts[1:])
    }


def text(split="train"):
    """Return unmodified real text for one disjoint split."""
    return _text_parts()[split][0]


def text_metadata(split="train"):
    content, start, end = _text_parts()[split]
    return dict(
        dataset="Tiny Shakespeare / char-rnn",
        source_url=TEXT_URL,
        raw_sha256=TEXT_SHA256,
        split=split,
        character_range=[start, end],
        split_sha256=hashlib.sha256(content.encode()).hexdigest(),
        split_method="contiguous 80/10/10, rounded to line boundaries before windows",
        vocab_size=len(vocabulary()),
        unknown_id=0,
        mask_id=0,
        vocabulary="training characters only; ID 0 is unknown/mask",
    )


@lru_cache(maxsize=1)
def vocabulary():
    """Indexable character string; index zero is unknown/mask."""
    return "\0" + "".join(sorted(set(text("train"))))


def encode(value):
    table = {c: i for i, c in enumerate(vocabulary())}
    return np.array([table.get(c, 0) for c in value], dtype=np.int64)


def decode(ids):
    vocab = vocabulary()
    return "".join(vocab[int(i)] if int(i) else "�" for i in ids)


def text_sequences(n=384, length=13, split="train", seed=17):
    """Sample complete windows within a split, never across its boundaries."""
    values = encode(text(split))
    if length < 1 or length > len(values):
        raise ValueError("Invalid text window length")
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(values) - length + 1, n, replace=n > len(values) - length + 1)
    return values[starts[:, None] + np.arange(length)]


def language_data(n=384, length=12, split="train", seed=17):
    seq = text_sequences(n, length + 1, split, seed)
    x, y = seq[:, :-1], seq[:, 1:]
    return x, y, np.ones_like(x, dtype=float)


def mnist_split(n=1024, split="train", side=7, seed=17, classes=None):
    """Return flattened images, original digit labels, and exact split provenance."""
    from . import mnist

    if side not in (7, 14, 28):
        raise ValueError("MNIST side must be 7, 14, or 28")
    path = Path(DEFAULT_CACHE) / "mnist.npz"
    download(mnist.URL, str(path), mnist.SHA256)
    if sha256_of(path) != mnist.SHA256:
        raise ValueError("MNIST checksum does not match the pinned dataset")
    with np.load(path, allow_pickle=False) as data:
        key = "test" if split == "test" else "train"
        labels = data["y_" + key]
        if split == "train":
            eligible = np.arange(50000)
        elif split == "validation":
            eligible = np.arange(50000, 60000)
        elif split == "test":
            eligible = np.arange(len(labels))
        else:
            raise ValueError("Unknown MNIST split")
        if classes is not None:
            eligible = eligible[np.isin(labels[eligible], classes)]
        rng = np.random.default_rng(seed)
        ids = rng.choice(eligible, min(n, len(eligible)), replace=False)
        images = data["x_" + key][ids].astype(float) / 255.0
        y = labels[ids].astype(int)
    k = 28 // side
    x = images.reshape(-1, side, k, side, k).mean((2, 4)).reshape(-1, side * side)
    meta = dict(
        dataset="MNIST",
        source_url=mnist.URL,
        raw_sha256=mnist.SHA256,
        split=split,
        official_partition=key,
        example_indices=ids.tolist(),
        preprocessing="pixels / 255; nonoverlapping mean pooling to %dx%d" % (side, side),
        seed=seed,
        examples=len(x),
        x_sha256=array_hash(x),
        y_sha256=array_hash(y),
    )
    return x, y, meta


def save_evidence(name, parameters, history, dataset, settings, source_files=()):
    """Save actual returned weights/traces with settings and their data identity.

    Callers must pass the settings that really produced these arrays. This is
    inspectable provenance, not proof that an arbitrary caller ran training.
    """
    arrays = {"parameter_" + k: np.asarray(v) for k, v in parameters.items()}
    arrays.update({"history_" + k: np.asarray(v) for k, v in history.items()})
    info = dict(
        dataset=dataset,
        settings=settings,
        numpy_version=np.__version__,
        source_sha256=source_hash("realdata.py", *source_files),
        arrays={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_hash(v))
            for k, v in arrays.items()
        },
    )
    run_id = hashlib.sha256(json.dumps(info, sort_keys=True, default=str).encode()).hexdigest()
    directory = Path(DEFAULT_CACHE) / "run-evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + "-" + run_id[:16] + ".npz")
    temporary = path.with_name(path.name + ".%d.part" % os.getpid())
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    info.update(run_id=run_id, artifact=str(path.resolve()), artifact_sha256=sha256_of(path))
    receipt = path.with_suffix(".json")
    temporary_receipt = receipt.with_name(receipt.name + ".%d.part" % os.getpid())
    temporary_receipt.write_text(json.dumps(info, indent=2, default=str) + "\n")
    os.replace(temporary_receipt, receipt)
    return info

"""Numerical contracts for the new transformer experiments."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import fd_check

from voxanim import tinylm as T
from voxanim.next_attention import _flash


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"norm": "rms"},
        {"placement": "post"},
        {"parallel": True},
        {"activation": "swiglu"},
        {"kv_heads": 1},
        {"tied": True},
        {"attention": "linear"},
        {"cross_attention": True},
        {"layers": 2},
        {"mask": "bidirectional"},
        {"mask": "prefix"},
        {"mask": "window"},
        {"mask": "sparse"},
        {"scale_scores": False},
    ],
)
def test_configurable_transformer_gradients(options):
    c = replace(T.Config(width=8, heads=2, kv_heads=2, ff=12, max_len=16), **options)
    P = T.init(c, 3)
    x = np.array([[1, 2, 3, 4], [3, 2, 1, 5]])
    y = (x + 2) % c.vocab
    context = x[:, ::-1].copy() if c.cross_attention else None
    weights = np.array([[0, 1, 1, 1], [1, 0, 1, 0]])
    _, G = T.loss_grads(P, x, y, c, weights, context=context)
    fd_check(lambda Q: T.loss(Q, x, y, c, weights, context=context), P, G, list(P), tol=3e-4)


def test_causal_mask_cannot_see_future():
    c = T.Config(width=8, heads=2, kv_heads=2, ff=12)
    P = T.init(c)
    x = np.array([[1, 2, 3, 4]])
    before = T.forward(P, x, c)
    x[:, 2:] = 7
    after = T.forward(P, x, c)
    np.testing.assert_allclose(before[:, :2], after[:, :2], atol=0, rtol=0)


def test_prefix_and_local_reach():
    c = T.Config(mask="prefix", prefix=3)
    m = T.attention_mask(8, c)
    assert m[:3, :3].all() and not m[:3, 3:].any()
    assert np.array_equal(m[3:], np.tri(8, dtype=bool)[3:])
    w = T.attention_mask(16, replace(c, mask="window", window=4))
    r = w.astype(int) @ w.astype(int) @ w.astype(int)
    assert r[-1, -10] > 0 and r[-1, -11] == 0


@pytest.mark.parametrize("tile", [1, 3, 8, 32])
def test_online_softmax_is_exact(tile):
    rng = np.random.default_rng(17)
    q, k = rng.normal(size=(23, 7)), rng.normal(size=(23, 7))
    v = rng.normal(size=(23, 5))
    s = q @ k.T / np.sqrt(7)
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    out, _ = _flash(q, k, v, tile)
    np.testing.assert_allclose(out, p @ v, atol=1e-14, rtol=1e-14)


def test_every_catalogue_entry_is_registered_once():
    from voxanim import presets
    from voxanim.next_catalog import ITEMS

    assert len(ITEMS) == 237            # the hundred of TODO.md and the groups added after
    assert len({v["preset"] for v in ITEMS}) == len(ITEMS)
    assert all(v["preset"] in presets.REGISTRY for v in ITEMS)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"layers": 3, "kv_heads": 1},
        {"norm": "rms", "activation": "swiglu"},
        {"placement": "post"},
        {"parallel": True},
        {"mask": "window"},
        {"mask": "sparse"},
        {"attention": "linear"},
        {"tied": True},
    ],
)
def test_tinylm_kv_cache_matches_full_forward(options):
    c = replace(T.Config(width=8, heads=2, kv_heads=2, ff=12), **options)
    P = T.init(c, 7)
    x = np.random.default_rng(8).integers(c.vocab, size=(2, 10))
    full = T.forward(P, x, c)
    cache, pieces = None, []
    for start, end in ((0, 3), (3, 4), (4, 7), (7, 10)):
        logits, cache = T.decode(P, x[:, start:end], c, cache)
        pieces.append(logits)
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), full, atol=1e-12, rtol=1e-12)
    assert cache.length == 10


def test_training_array_identity_includes_shape_and_dtype():
    flat = np.arange(12, dtype=np.int64)
    assert T.array_identity(flat.reshape(2, 6)) != T.array_identity(flat.reshape(3, 4))
    assert T.array_identity(flat) != T.array_identity(flat.view(np.float64))


def test_training_records_checkpoint_and_actual_updates(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    c = T.Config(width=8, heads=2, kv_heads=2, ff=12, max_len=8)
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    y = (x + 1) % c.vocab
    result = T.train(c, x, y, steps=4, name="evidence-test", cache_dir=tmp_path)
    record = result["provenance"]
    assert record["actual_updates"] == record["checkpoint_step"] == 4
    assert record["loss_update_numbers"] == [1, 2, 3, 4]
    assert record["final_parameter_sha256"] == T.parameter_hash(result["P"])
    assert record["initial_parameter_sha256"] != record["final_parameter_sha256"]
    assert record["settings"]["x"]["shape"] == [2, 4]
    assert record["settings"]["sources"]["tinylm.py"]
    artifact = Path(record["artifact"]["path"])
    assert record["artifact"]["sha256"] == T.sha256_of(artifact)
    with np.load(artifact, allow_pickle=False) as stored:
        assert json.loads(str(stored["run_manifest"]))["actual_updates"] == 4
        assert np.array_equal(stored["P_E"], result["P"]["E"])

    def fail_if_training_runs(*args, **kwargs):
        raise AssertionError("A matching verified checkpoint should be reused")

    monkeypatch.setattr(T, "loss_grads", fail_if_training_runs)
    cached = T.train(c, x, y, steps=4, name="evidence-test", cache_dir=tmp_path)
    assert cached["provenance"] == record
    assert np.array_equal(cached["loss"], result["loss"])


def test_training_source_and_data_changes_invalidate_cache(tmp_path, monkeypatch):
    c = T.Config(width=8, heads=2, kv_heads=2, ff=12, max_len=8)
    x = np.arange(12).reshape(2, 6) % c.vocab
    y = (x + 1) % c.vocab
    first = T.train(c, x, y, steps=1, name="identity-test", cache_dir=tmp_path)
    reshaped = T.train(
        c, x.reshape(3, 4), y.reshape(3, 4), steps=1, name="identity-test", cache_dir=tmp_path
    )
    assert first["provenance"]["artifact"]["path"] != reshaped["provenance"]["artifact"]["path"]
    sources = T.training_sources()
    monkeypatch.setattr(T, "training_sources", lambda: {**sources, "tinylm.py": "changed"})
    changed = T.train(c, x, y, steps=1, name="identity-test", cache_dir=tmp_path)
    assert first["provenance"]["artifact"]["path"] != changed["provenance"]["artifact"]["path"]


def test_real_text_tasks_preserve_observed_targets():
    from pathlib import Path

    from voxanim import next_attention as attention
    from voxanim import realdata
    from voxanim.cache import DEFAULT_CACHE

    if not (Path(DEFAULT_CACHE) / "tinyshakespeare.txt").exists():
        pytest.skip("Pinned Tiny Shakespeare dataset is not cached")
    for split in ("train", "test"):
        expected = realdata.text_sequences(n=8, length=12, split=split, seed=31)
        for task in ("fill", "copy"):
            x, y, weights = attention.text_task(task, n=8, seed=31, split=split)
            np.testing.assert_array_equal(y, expected)
            assert np.all(x[weights > 0] == realdata.text_metadata(split)["mask_id"])
            np.testing.assert_array_equal(x[weights == 0], expected[weights == 0])
        x, y, weights = attention.text_task(n=8, seed=31, split=split)
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
        assert np.all(weights == 1)
    train = realdata.text_metadata("train")["character_range"]
    test = realdata.text_metadata("test")["character_range"]
    assert train[1] <= test[0]


def test_real_text_denoising_inputs_and_checkpoint_replay(tmp_path):
    from pathlib import Path

    from voxanim import next_attention as attention
    from voxanim import realdata
    from voxanim.cache import DEFAULT_CACHE

    if not (Path(DEFAULT_CACHE) / "tinyshakespeare.txt").exists():
        pytest.skip("Pinned Tiny Shakespeare dataset is not cached")
    source, decoder, target = attention.denoising_data(8, "train", 31)
    np.testing.assert_array_equal(target, realdata.text_sequences(8, 6, "train", 31))
    assert np.all(source[:, 1::3] == 0) and np.all(decoder[:, 0] == 0)
    np.testing.assert_array_equal(decoder[:, 1:], target[:, :-1])
    c = T.Config(vocab=len(realdata.vocabulary()), width=8, heads=2, kv_heads=2, ff=12)
    x, y, w = attention.text_task(n=8, length=6)
    run = T.train(
        c,
        x,
        y,
        weights=w,
        steps=2,
        name="real-text-replay",
        cache_dir=tmp_path,
        provenance=attention.task_provenance(),
    )
    xt, yt, wt = attention.text_task(n=8, length=6, split="test")
    measured = T.loss(run["P"], xt, yt, c, wt)
    with np.load(run["provenance"]["artifact"]["path"], allow_pickle=False) as archive:
        parameters = {k[2:]: archive[k] for k in archive.files if k.startswith("P_")}
    assert measured == T.loss(parameters, xt, yt, c, wt)
    assert (
        run["provenance"]["settings"]["provenance"]["dataset"]["raw_sha256"] == realdata.TEXT_SHA256
    )

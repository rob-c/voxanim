"""End-to-end contracts for the local, unpublished review catalogue.

All studies use their normal seeded caches. Dataset-dependent cases inherit the
same skip marks as the public preset registry, including collected real text.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import data_marks, needs_text

from voxanim import accel, presets
from voxanim.mech import Mechanism
from voxanim.next_attention import counts, make, sparse_mask
from voxanim.next_catalog import GPT2, ITEMS, MNIST, NO_DATA, TEXT, build_study, drawing
from voxanim.scene import Renderer, Style
from voxanim.tinylm import Config, init

CATALOGUE = [
    pytest.param(item, id=item["preset"], marks=data_marks(item["preset"])) for item in ITEMS
]


@pytest.mark.parametrize("item", CATALOGUE)
def test_catalogue_has_measured_finite_panels(item):
    study = build_study(item["module"], item["name"]).validate()
    assert len(study.panels) == 4
    assert study.title and study.subtitle
    for panel in study.panels:
        assert panel.title and panel.caption
        data = panel.data
        if panel.kind == "curve":
            x, ys = np.asarray(data["x"]), np.atleast_2d(data["ys"])
            assert x.ndim == 1 and len(x) > 0
            assert ys.shape[1] == len(x)
            assert len(data["labels"]) == len(ys)
            assert np.isfinite(x).all() and np.isfinite(ys).all()
        elif panel.kind == "bars":
            values = np.asarray(data["values"])
            assert values.ndim == 1 and len(values) > 0
            assert len(data["labels"]) == len(values)
            assert np.isfinite(values).all()
        elif panel.kind == "matrix":
            values = np.asarray(data["values"])
            assert values.ndim == 2 and values.size > 0
            assert np.isfinite(values).all()
        elif panel.kind == "points":
            xy = np.asarray(data["xy"])
            assert xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0
            assert np.isfinite(xy).all()
            if "classes" in data:
                assert len(data["classes"]) == len(xy)
        else:
            images = np.asarray(data["values"])
            assert images.ndim in (3, 4) and len(images) > 0
            assert np.isfinite(images).all()
            if "labels" in data:
                assert len(data["labels"]) == len(images)
    # ... and is drawn as a mechanism in named phases, not as its four panels
    anim = drawing(item["name"])(study)
    assert isinstance(anim, Mechanism)
    assert len(anim.PHASES) >= 4
    # one is deliberately long: a training run followed on a real clock, to set expectations
    longest = 180.0 if item["name"] == "cpu-and-gpu-training" else 66.0
    assert 45.0 <= anim.duration <= longest
    for k, (start, end) in enumerate(anim.spans):
        assert anim.subtitle_for(k).strip()
        scene = anim.scene(start + 0.6 * (end - start))
        assert sum(t.opacity > 0.5 and bool(t.s.strip()) for t in scene.texts) >= 3


def test_registry_classifies_dataset_dependencies(monkeypatch):
    import conftest

    monkeypatch.setattr(conftest, "gpt2_cached", lambda: False)
    monkeypatch.setattr(conftest, "mnist_cached", lambda: False)
    monkeypatch.setattr(conftest, "text_cached", lambda: False)
    for item in ITEMS:
        name = item["name"]
        preset = item["preset"]
        assert preset in presets.REGISTRY
        expected = []
        if name in GPT2:
            expected.append(conftest.needs_gpt2)
        if name in MNIST:
            expected.append(conftest.needs_mnist)
        if name in TEXT:
            expected.append(conftest.needs_text)
        assert data_marks(preset) == expected
        if expected:
            assert preset not in NO_DATA
            assert not conftest.available(preset)
        else:
            assert preset in NO_DATA
            assert data_marks(preset) == []
            assert conftest.available(preset)


LOOPED = [pytest.param(name, marks=data_marks(name), id=name)
          for name in ("causal-vs-bidirectional", "data-parallel-allreduce", "mcts-tic-tac-toe")]


@pytest.mark.parametrize("name", LOOPED)
def test_a_mechanism_renders_and_loops_exactly(name):
    from voxanim import presets
    anim = presets.build(name)
    style = Style(width=480, height=270, ss=1, motion_blur=False, reflection=0.0)
    with accel.forced("numpy"):
        renderer = Renderer(anim.cam, style)
        initial = renderer.render(anim.scene(0.0))
        np.testing.assert_array_equal(initial, renderer.render(anim.scene(anim.duration)))
        for start, end in anim.spans:
            time = start + 0.6 * (end - start)
            frame = renderer.render(anim.scene(time))
            assert frame.shape == (270, 480, 3)
            assert np.isfinite(frame).all()
            assert np.max(np.abs(frame - initial)) > 0.01
            np.testing.assert_array_equal(frame, renderer.render(anim.scene(time)))


@needs_text
def test_bidirectional_fill_reports_real_text_outcomes():
    study = make("causal-vs-bidirectional")
    causal, bidirectional = study.metadata["heldout_loss"]
    assert np.isfinite([causal, bidirectional]).all()
    assert causal >= 0 and bidirectional >= 0
    assert "Shakespeare" in str(study.metadata)


@needs_text
def test_local_depth_reach_is_distinct_from_measured_accuracy():
    study = make("sliding-window-attention")
    one_layer, two_layers, _three_layers = study.metadata["heldout_loss"]
    assert np.isfinite([one_layer, two_layers, _three_layers]).all()
    # The structural claim is independent of how any run trains.
    from voxanim.tinylm import attention_mask

    local = attention_mask(12, Config(mask="window", window=4))
    assert not local[9, 3]
    assert (local.astype(int) @ local.astype(int))[9, 3] > 0


@needs_text
def test_encoder_decoder_measures_real_text_restoration():
    study = make("encoder-decoder-family")
    loss = study.panels[-1].data["values"]
    assert len(loss) == 3
    assert np.isfinite(loss).all() and np.all(loss >= 0)
    # The third entry is a jointly trained encoder and decoder, not a prefix mask.
    assert "Joint target accuracy" in study.panels[-1].notes[0]
    assert "contextual hidden states" in study.panels[2].caption


def test_tied_embeddings_save_exact_output_matrix():
    untied = Config(vocab=13, width=16, heads=4, kv_heads=4, ff=32, max_len=16)
    tied = replace(untied, tied=True)
    up, tp = init(untied, 8), init(tied, 8)
    assert counts(up) - counts(tp) == untied.vocab * untied.width
    assert "U" in up and "U" not in tp
    assert up["E"].shape == tp["E"].shape == (13, 16)


@pytest.mark.parametrize("kind", ["dense", "strided", "block", "global", "random"])
def test_sparse_patterns_are_seeded_and_causal(kind):
    mask = sparse_mask(40, kind, 4)
    np.testing.assert_array_equal(mask, sparse_mask(40, kind, 4))
    assert np.diag(mask).all()
    assert not np.triu(mask, 1).any()
    if kind == "block":
        assert not mask[39, :36].any()
    if kind in ("global", "random"):
        assert mask[:, 0].all()

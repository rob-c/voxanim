"""The encoder-decoder with attention (:mod:`voxanim.seq2seq`) and its
animation: gradients, the task is learned, the alignment is the anti-diagonal,
and the subtitles' numbers."""

from __future__ import annotations

import numpy as np
from conftest import fd_check

from voxanim import seq2seq as S


def test_gradients_match_finite_differences():
    """Every gradient of the model (GRU cells, attention, embeddings, output)."""
    rng = np.random.default_rng(3)
    P = S.init(rng, n=5, d=3, m=4)
    x, y = S.reverse_batch(rng, 3, 4)
    y_in = S.teacher_inputs(y)
    g = S.backward(P, S.forward(P, x, y_in), y)
    fd_check(lambda Q: S.loss(S.forward(Q, x, y_in), y), P, g, list(P))


def test_attention_rows_are_distributions():
    rng = np.random.default_rng(0)
    P = S.init(rng)
    x, y = S.reverse_batch(rng, 4)
    f = S.forward(P, x, S.teacher_inputs(y))
    assert np.allclose(f["alpha"].sum(-1), 1.0) and (f["alpha"] >= 0).all()
    assert np.allclose(f["p"].sum(-1), 1.0)
    # the context is the attention-weighted mix of the input embeddings
    c = np.einsum("njl,nld->njd", f["alpha"], P["E"][x])
    assert np.allclose(c, f["c"])


def test_task_is_learned_and_the_alignment_is_the_anti_diagonal():
    """The trained model reverses every held-out sequence, and each output
    token attends to the input token it copies."""
    tr = S.trace()
    run = tr.run
    assert np.array_equal(tr.pred, tr.y)
    assert run.accuracy[-1] == 1.0
    assert S.token_accuracy(run.P, *S.held_out(200, seed=99)) == 1.0
    assert list(tr.alpha.argmax(1)) == [S.L - 1 - j for j in range(S.L)]
    anti = np.mean([tr.alpha[j, S.L - 1 - j] for j in range(S.L)])
    assert anti > 0.8
    # ... and not only for the example: on average over held-out sequences
    x, _ = S.held_out(200)
    mean_alpha = S.decode(run.P, x)[1].mean(0)
    assert list(mean_alpha.argmax(1)) == [S.L - 1 - j for j in range(S.L)]


def test_untrained_alignment_is_flat_and_the_run_is_consistent():
    """Before training the attention is spread over the inputs and the output
    is wrong; the recorded run agrees with the weights it stores."""
    run = S.trace().run
    a0 = run.alpha[0]
    assert a0.max() < 0.25 and a0.min() > 0.1          # about 1/6 everywhere
    assert run.accuracy[0] < 0.2                         # chance is 0.1
    assert run.alpha.shape == (S.STEPS + 1, S.L, S.L)
    assert run.eval_steps[0] == 0 and run.eval_steps[-1] == S.STEPS
    x, _ = S.example()
    pred, alpha = S.decode(run.P, x[None])
    assert np.array_equal(pred[0], run.pred[-1])
    assert np.allclose(alpha[0], run.alpha[-1], atol=1e-5)
    assert run.losses[-100:].mean() < 0.01


def test_animation_claims():
    """The numbers the subtitles quote: accuracy at the start and the end."""
    from voxanim.seq2seq_anim import Seq2SeqAttention
    anim = Seq2SeqAttention(S.trace())
    run = anim.tr.run
    assert run.accuracy_at(0) < 0.2 and run.accuracy_at(S.STEPS) == 1.0
    assert 40 <= anim.duration <= 60
    sc = anim.scene(anim.Tm["decode"][1] + 0.1)
    assert sc.voxels and sc.texts

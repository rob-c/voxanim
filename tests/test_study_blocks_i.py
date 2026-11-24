"""The numbers behind animations 209-211 (voxanim.study_blocks_i): the held-out count and
accuracy, the state_dict saved and loaded, and one recorded training step."""

from __future__ import annotations

import math

import numpy as np
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import study_blocks_i as I
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_correct_is_a_running_total_over_the_test_batches():
    right = np.zeros(10_000, bool)
    right[::2] = True                       # every other digit right
    run = I.running_correct(right)
    assert len(run) == 10 and list(run[:3]) == [500, 1000, 1500] and run[-1] == 5000
    np.testing.assert_array_equal(np.diff(np.r_[0, run]), right.reshape(10, -1).sum(1))


def test_the_corners_drawn_of_each_tensor():
    w1 = np.arange(64 * 784).reshape(64, 784)
    c = I.corner("1.weight", w1)
    assert c.shape == (8, 8)
    assert c[0, 0] == w1[0, 14 * 28 + 10] and c[7, 7] == w1[7, 14 * 28 + 17]
    w5 = np.arange(320).reshape(10, 32)
    np.testing.assert_array_equal(I.corner("5.weight", w5), w5[:8, :8])
    np.testing.assert_array_equal(I.corner("1.bias", np.arange(64)), np.arange(8))


def test_the_settings_are_the_scripts():
    assert str(I.TEST_BATCH) in B.code_line("loader", "test_loader")
    assert '"%s"' % I.FILE in B.code_line("save", "torch.save")
    assert '"%s"' % I.FILE in B.code_line("load", "torch.load")
    assert len(B.code_lines("step")) == 6 and len(B.code_lines("evaluate")) == 8


# ---------------------------------------------------------------------- 209
@needs_mnist
def test_evaluation_counts_the_held_out_digits():
    s = I.evaluation().validate()
    m = s.metadata
    yt = course.data()[3]
    assert m["test"] == 10_000 and m["batches"] == 10
    assert sum(m["per_batch"]) == m["correct"] == m["running"][-1]
    assert m["running"] == list(np.cumsum(m["per_batch"]))
    assert m["correct"] + m["wrong"] == 10_000
    assert math.isclose(m["accuracy"], m["correct"] / 10_000)
    assert m["accuracy"] == course.heldout("sgd")["accuracy"][-1] == course.summary()["sgd"][
        "accuracy"]
    assert 0.94 < m["accuracy"] < 0.96 and m["accuracy_text"] == "%.4f" % m["accuracy"]
    assert m["printed"] == str(m["correct"] / 10_000)        # what the print line writes
    # the first ten, one by one: their labels are the test set's, the mistake is a mistake
    assert m["shown_labels"] == [int(v) for v in yt[:I.SHOWN]]
    assert m["shown_correct"] == sum(m["shown_right"]) == I.SHOWN - 1
    w = m["first_wrong"]
    assert not m["shown_right"][w] and all(m["shown_right"][:w])
    assert m["first_wrong_label"] == yt[w] != m["first_wrong_pred"] == m["shown_preds"][w]
    assert [p == y for p, y in zip(m["shown_preds"], m["shown_labels"])] == m["shown_right"]
    # no_grad: no gradient made, where a training step makes one per parameter
    assert m["grads_made"] == 0 and m["grads_training"] == course.count() == 52_650
    assert m["dropout_eval"] > m["dropout_train_mode"]           # where eval() matters
    assert m["code"] == list(B.code_lines("evaluate")) and m["train_mode"] == "model.train()"


# ---------------------------------------------------------------------- 210
@needs_mnist
def test_save_and_load_keeps_every_number_and_every_answer():
    s = I.save_and_load().validate()
    m = s.metadata
    assert m["keys"] == ["1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias"]
    assert m["shapes"] == [[64, 784], [64], [32, 64], [32], [10, 32], [10]]
    assert m["sizes"] == [int(np.prod(sh)) for sh in m["shapes"]]
    assert sum(m["sizes"]) == m["numbers"] == 52_650 and m["numbers_text"] == "52,650"
    assert m["bytes"] == 4 * 52_650 == 210_600
    assert m["max_change"] == 0.0 and m["same"] == m["test"] == 10_000
    assert m["accuracy_before"] == m["accuracy_after"] == course.heldout("sgd")["accuracy"][-1]
    P = course.params(course.EPOCH_STEPS)
    for k in m["keys"]:
        np.testing.assert_array_equal(np.asarray(m["corners"][k]), I.corner(k, P[k]))
        assert not np.allclose(m["fresh"][k], m["corners"][k])  # the new model differs
    xy = np.asarray(s.panels[2].data["xy"])
    assert len(xy) == 10_000 and (xy[:, 0] == xy[:, 1]).all()


# ---------------------------------------------------------------------- 211
@needs_mnist
def test_one_step_is_the_recorded_step():
    s = I.one_step().validate()
    m = s.metadata
    assert m["step"] == I.STEP and m["number"] == I.STEP + 1 and m["steps"] == 938
    b = course.batch(I.STEP)
    assert m["labels"] == [int(v) for v in b["y"]] and m["batch"] == 64
    assert m["shapes"] == ["[64, 1, 28, 28]", "[64, 784]", "[64, 64]", "[64, 32]", "[64, 10]"]
    assert math.isclose(m["loss"], course.losses()[I.STEP], rel_tol=1e-9)
    assert abs(m["loss"] - math.log(10)) < 0.05                 # still guessing
    assert m["loss_after"] < m["loss"]                         # the step helped this batch
    assert int(np.argmax(m["logits"])) == m["first_pred"]
    w = m["weight"]
    i, j = w["index"]
    assert w["before"] == course.params(I.STEP)[w["key"]][i, j]
    assert w["grad"] == course.grads(I.STEP)[w["key"]][i, j]
    assert w["after"] == course.params(I.STEP + 1)[w["key"]][i, j] == w["check"]
    assert abs(w["grad"]) == np.abs(np.asarray(m["grads"][w["key"]])).max()
    assert m["grad_shape"] == "[64, 784]" == B.shape(course.SIZES[1::-1])
    assert m["code"] == list(B.code_lines("step"))

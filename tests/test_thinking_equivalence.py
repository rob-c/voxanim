"""examples/thinking.py's own model, in PyTorch, from voxanim.study_thinking's starting weights
on its first batch: the same loss, gradients and Adam step (to 1e-5); and its examples and
greedy generation are the twin's."""

from __future__ import annotations

import runpy

import numpy as np
import pytest
from test_mnist_blocks_equivalence import close

from voxanim import blocks
from voxanim import study_thinking as T

torch = pytest.importorskip("torch", reason="torch is not installed: the thinking script is "
                                            "checked with it")


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(blocks.THINKING), run_name="test")


def test_examples_match(ns):
    every, _held = T.strings()
    for e in range(len(T.EFFORTS)):
        bits = every[1234].tolist()
        assert ns["example"](bits, e) == T.example(bits, e)


def test_one_step_matches_the_twin(ns):
    m = ns["model"]
    assert list(m.state_dict()) == list(T.keys())
    assert sum(p.numel() for p in m.parameters()) == T.count()
    m.load_state_dict({k: torch.tensor(v) for k, v in T.init_params().items()})
    b = T.batch(0)
    x, learn = torch.tensor(b["x"]), torch.tensor(b["learn"])
    run = dict(ns, x=x, learn=learn, optimizer=torch.optim.Adam(m.parameters(), lr=T.LR))
    exec("\n".join(blocks.code_lines("th-train")), run)
    loss, _G = T.loss_and_grads(T.init_params(), b["x"], b["learn"])
    assert abs(run["loss"].item() - loss) < 1e-5
    W1 = T.params(1)
    for k, v in m.state_dict().items():
        # Adam's first step moves a weight by about lr whatever its gradient's size; the key
        # bias's gradient is zero in exact arithmetic (softmax ignores a shift shared by all
        # scores), so rounding decides its step: compare where the gradient is not next to 0
        real = np.abs(T.run()["G0__" + k]) > 1e-6
        close(v.numpy()[real], W1[k][real])
        assert np.abs(v.numpy() - W1[k]).max() <= 2 * T.LR


def test_generation_matches_the_twin(ns):
    m = ns["model"]
    m.load_state_dict({k: torch.tensor(v) for k, v in T.params().items()})
    every, held = T.strings()
    for e in (0, 3, 6):
        mine = T.generate(T.params(), every[held[:20]], e)
        for row, bits in zip(mine, every[held[:20]], strict=True):
            assert ns["answer"](bits.tolist(), e) == row.tolist()

"""voxanim.study_thinking: the task, the gradients, and the measured claims."""

from __future__ import annotations

import numpy as np

from voxanim import study_thinking as T


def test_the_task():
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    assert T.example(bits, 0) == [3] + bits + [2, 1]                 # 7 ones: odd
    assert T.example(bits, 6)[14:] == [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    assert T.example(bits, 3)[14:] == [1, 0, 1, 1]                   # after 4, 8, 12 bits
    every, held = T.strings()
    assert every.shape == (4096, 12) and len(held) == 468 and len(set(held)) == 468
    b = T.batch(0)
    assert b["x"].shape == (256, 27) and not np.isin(b["x"][:, 1:13], [2]).any()


def test_the_gradients_against_finite_differences():
    rng = np.random.default_rng(5)
    P = {k: v.astype(np.float64) for k, v in T.init_params().items()}
    b = T.batch(1)
    x, learn = b["x"][:4], b["learn"][:4]
    _l, G = T.loss_and_grads(P, x, learn)
    for k in ("tok.weight", "blocks.0.qkv.weight", "blocks.1.mlp.0.weight", "head.bias",
              "blocks.1.n1.weight"):
        idx = tuple(rng.integers(0, s) for s in P[k].shape)
        if k == "tok.weight":
            idx = (int(x[0, 0]), 3)
        up, down = {**P, k: P[k].copy()}, {**P, k: P[k].copy()}
        up[k][idx] += 1e-6
        down[k][idx] -= 1e-6
        num = (T.loss_and_grads(up, x, learn)[0] - T.loss_and_grads(down, x, learn)[0]) / 2e-6
        assert abs(num - G[k][idx]) < 1e-5 * max(1.0, abs(num)), (k, num, G[k][idx])


def test_the_results():
    r = T.results()
    assert [x["effort"] for x in r] == list(T.EFFORTS)
    assert [x["tokens"] for x in r] == [k + 1 for k in T.EFFORTS]
    assert r[-1]["right"] >= 460                                     # a step per bit: right
    assert r[0]["right"] < 0.6 * 468                                 # no steps: about chance
    assert T.run()["loss"][-100:].mean() < T.run()["loss"][0] / 5

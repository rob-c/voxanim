"""voxanim.study_aegan: the blocks, the rewiring and the measured claims."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_aegan as A


def test_grey_and_shapes():
    assert A.grey(np.array([0.0, 0.1, 0.5, 0.9, 1.0])) == 0.2
    e, d = A.init()
    assert e["0.weight"].shape == (256, 784) and e["2.weight"].shape == (16, 256)
    assert d["0.weight"].shape == (256, 16) and d["2.weight"].shape == (784, 256)
    assert A.judge_head()["2.weight"].shape == (1, 256)


@needs_mnist
def test_the_rewiring_keeps_the_blocks():
    enc, dec, j0 = A.blocks("enc"), A.blocks("dec"), A.blocks("judge0")
    assert np.array_equal(j0["0.weight"], enc["0.weight"]) and np.array_equal(j0["0.bias"],
                                                                              enc["0.bias"])
    assert j0["2.weight"].shape == (1, 256)                  # the only new layer
    gen1 = A.blocks("gen1")
    assert all(gen1[k].shape == dec[k].shape for k in dec)


@needs_mnist
def test_the_measured_claims():
    r = A.results()
    assert r["real"]["grey"] < r["after"]["grey"] < r["before"]["grey"]   # sharper after
    assert r["after"]["grey"] < r["rebuilt"]["grey"]
    assert r["after"]["sure"] > r["before"]["sure"]
    assert min(r["after"]["counts"]) > 20                    # every digit made
    assert r["rebuild_right"] > 0.9
    assert len(A.run()["lj"]) == A.GAN_STEPS and len(A.run()["ae_loss"]) == A.AE_STEPS

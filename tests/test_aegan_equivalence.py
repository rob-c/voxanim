"""examples/ae_gan.py's own lines, in PyTorch, from voxanim.study_aegan's starting weights on its
first batches: one autoencoder step and one GAN step give the same losses and gradients, and the
rewiring keeps the encoder's body and the decoder exactly."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist
from test_mnist_blocks_equivalence import close

from voxanim import blocks
from voxanim import study_aegan as A

torch = pytest.importorskip("torch", reason="torch is not installed: the AE-to-GAN script is "
                                            "checked with it")
F = torch.nn.functional


def run(regions, ns):
    for r in regions:
        exec("\n".join(blocks.code_lines(r)), ns)
    return ns


def built():
    ns = run(("ag-blocks",), dict(torch=torch, nn=torch.nn, F=F))
    assert list(ns["encoder"].state_dict()) == list(A.ENC_KEYS)
    assert list(ns["decoder"].state_dict()) == list(A.DEC_KEYS)
    return ns


@needs_mnist
def test_one_autoencoder_step():
    ns = built()
    e0, d0 = A.init()
    ns["encoder"].load_state_dict({k: torch.tensor(v) for k, v in e0.items()})
    ns["decoder"].load_state_dict({k: torch.tensor(v) for k, v in d0.items()})
    ns["x"] = torch.tensor(A.ae_batch(0))
    run(("ag-ae",), ns)
    ns["loss"].backward()
    assert abs(ns["loss"].item() - A.run()["ae_loss"][0]) < 1e-5
    for k, p in ns["encoder"].named_parameters():
        close(p.grad.numpy(), A.run()["aeG0e__" + k])
    for k, p in ns["decoder"].named_parameters():
        close(p.grad.numpy(), A.run()["aeG0d__" + k])


@needs_mnist
def test_the_rewiring_and_one_gan_step():
    ns = built()
    enc, dec = A.blocks("enc"), A.blocks("dec")
    ns["encoder"].load_state_dict({k: torch.tensor(v) for k, v in enc.items()})
    ns["decoder"].load_state_dict({k: torch.tensor(v) for k, v in dec.items()})
    run(("ag-rewire",), ns)
    gen, judge = ns["generator"], ns["judge"]
    assert gen is ns["decoder"]                               # the same block
    assert judge[0] is ns["encoder"][0]                       # the encoder's body, as it is
    j0 = A.blocks("judge0")
    with torch.no_grad():
        judge[2].weight.copy_(torch.tensor(j0["2.weight"]))
        judge[2].bias.copy_(torch.tensor(j0["2.bias"]))
    np.testing.assert_array_equal(judge[0].weight.detach().numpy(), enc["0.weight"])
    x, z = A.gan_batch(0, A.run()["mean"], A.run()["spread"])
    ns.update(x=torch.tensor(x), mean=torch.zeros(16), spread=torch.ones(16),
              ones=torch.ones(64, 1), zeros=torch.zeros(64, 1))
    ns["torch"] = type("T", (), {"randn": staticmethod(lambda *s: torch.tensor(z))})
    run(("ag-judge",), ns)
    ns["lj"].backward()
    assert abs(ns["lj"].item() - A.run()["lj"][0]) < 1e-5
    for k, p in judge.named_parameters():
        close(p.grad.numpy(), A.run()["ganGj__" + k])

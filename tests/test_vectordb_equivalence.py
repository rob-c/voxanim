"""examples/vector_db.py's own lines, run in PyTorch, give voxanim.study_vectordb's numbers: one
InfoNCE step with Adam (loss, gradients, new weights, to 1e-5), the stored vectors and an exact
search, the IVF lists and candidates, and the mixed-encoder search. The script's random
batches, pairs and k-means start are replaced by the twin's (its generator is numpy's)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist
from test_mnist_blocks_equivalence import close, run_script

from voxanim import study_course as S
from voxanim import study_vectordb as D

torch = pytest.importorskip("torch", reason="torch is not installed: the vector-database "
                                            "script is checked with it")
F = torch.nn.functional


def encoder_with(P):
    ns = run_script(("vdb-encoder",), dict(torch=torch, nn=torch.nn))
    enc = ns["encoder"]
    assert tuple(enc.state_dict()) == D.KEYS
    enc.load_state_dict({k: torch.tensor(v) for k, v in P.items()})
    return enc


@needs_mnist
def test_one_infonce_step_matches_the_twin():
    enc = encoder_with(D.init_params(0))
    assert sum(p.numel() for p in enc.parameters()) == D.count() == 104_608
    b = D.batch(0, 0)
    ns = dict(torch=torch, F=F, encoder=enc, x=torch.tensor(b["x"]).reshape(-1, 1, 28, 28),
              y=torch.tensor(b["y"]),
              partner=lambda y: torch.tensor(b["p"]).reshape(-1, 1, 28, 28),
              optimizer=torch.optim.Adam(enc.parameters()))
    run_script(("vdb-loss", "vdb-step"), ns)
    assert abs(ns["loss"].item() - D.history(0)["loss"][0]) < 1e-5
    for k, p in enc.named_parameters():
        close(p.grad.numpy(), D.first_grads(0)[k])
    for k, v in enc.state_dict().items():
        close(v.numpy(), D.params(0, "after_one_step")[k])


@needs_mnist
def test_store_search_ivf_and_versions():
    x, y, xt, _yt = S.data()
    enc = encoder_with(D.params(0))
    ns = dict(torch=torch, F=F, encoder=enc, X=torch.tensor(x).reshape(-1, 1, 28, 28),
              Y=torch.tensor(y), XT=torch.tensor(xt).reshape(-1, 1, 28, 28))
    run_script(("vdb-store", "vdb-search"), ns)
    close(ns["store"].numpy(), D.store(0)["vectors"])
    want = D.search(0)
    assert list(ns["found"].numpy()) == list(want["ids"])
    assert list(ns["labels"][ns["found"]].numpy()) == list(want["labels"])
    iv = D.ivf()
    ns["kmeans"] = lambda v, k: torch.tensor(iv["centres"])
    run_script(("vdb-ivf-build", "vdb-ivf-search"), ns)
    assert np.array_equal(ns["home"].numpy(), iv["home"])
    near = np.argsort(-(iv["centres"] @ D.query(0)))[:4]
    assert len(ns["cand"]) == int(np.isin(iv["home"], near).sum())
    ns["train"] = lambda seed: encoder_with(D.params(seed))
    run_script(("vdb-versions",), ns)
    mixed = D.search(0, seed=1, store_seed=0)
    assert list(ns["labels"][ns["scores"].topk(5).indices].numpy()) == list(mixed["labels"])
    run_script(("vdb-reembed",), ns)
    again = D.search(0, seed=1, store_seed=1)
    assert list(ns["labels"][ns["scores"].topk(5).indices].numpy()) == list(again["labels"])

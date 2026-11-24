"""examples/padding.py's own lines, run in PyTorch, give the numbers voxanim.study_padding keeps
for the padding animations: the shapes, F.pad's four modes, the padded CNN's first SGD step
(to 1e-5), torch.stack's error, the padded block, the padded lines and their mask, the
masked mean and the attention mask. Skipped without torch, MNIST or the text."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from conftest import needs_mnist, needs_text
from test_mnist_blocks_equivalence import close, run_script

from voxanim import study_course as S
from voxanim import study_padding as P

torch = pytest.importorskip("torch", reason="torch is not installed: the padding script is "
                                            "checked with it")
F = torch.nn.functional


def namespace():
    from torch.nn.utils.rnn import pad_sequence
    x = torch.tensor(np.stack([S.digit(i)[0] for i in range(64)]))   # the first 64 digits
    return dict(torch=torch, nn=torch.nn, F=F, pad_sequence=pad_sequence, x=x)


def printed(regions, ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_script(regions, ns)
    return out.getvalue()


@needs_mnist
def test_the_shapes():
    torch.manual_seed(0)
    ns = namespace()
    run_script(("pad-none",), ns)
    assert tuple(ns["maps"].shape) == (64, 8, 26, 26)
    run_script(("pad-one",), ns)
    assert tuple(ns["maps"].shape) == (64, 8, 28, 28)
    out = printed(("pad-same",), ns).splitlines()
    for row, line in zip([r for r in P.same_and_valid() for _ in (0, 1)], out, strict=True):
        assert line.startswith("%d torch.Size([64, 8, " % row["k"])
    sides = [int(s.split(", ")[-1].rstrip("])")) for s in out]
    assert sides == [v for r in P.same_and_valid() for v in (r["same"], r["valid"])]
    assert printed(("pad-deep",), ns).strip() == P.PRINTED_DEEP
    assert P.deep()["valid"][-1] == 4


@needs_mnist
def test_the_four_modes():
    ns = namespace()
    out = printed(("pad-modes",), ns)
    assert out.splitlines() == ["%s torch.Size([1, 1, 12, 12])" % m for m in P.MODES]
    part = ns["x"][:1, :, P.PART[0], P.PART[1]]
    close(part[0, 0].numpy(), P.part())
    for mode, want in P.modes().items():
        close(F.pad(part, [2] * 4, mode=mode)[0, 0].numpy(), want)


@needs_mnist
def test_the_padded_cnn_takes_the_twins_first_step():
    ns = run_script(("pad-model",), namespace())
    model = ns["model"]
    assert tuple(model.state_dict()) == P.PAD_KEYS
    assert sum(p.numel() for p in model.parameters()) == P.count() == 9098
    model.load_state_dict({k: torch.tensor(v) for k, v in P.init_params().items()})
    b = S.batch(0)
    x, y = torch.tensor(b["x"]), torch.tensor(b["y"])
    with torch.no_grad():
        close(model(x).numpy(), P.forward(P.init_params(), b["x"])["logits"])
    opt = torch.optim.SGD(model.parameters(), lr=S.LR)
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    opt.zero_grad()
    loss.backward()
    assert abs(loss.item() - P.run()["loss"][0]) < 1e-5
    for k, p in model.named_parameters():
        close(p.grad.numpy(), P.first_grads()[k])
    opt.step()
    for k, v in model.state_dict().items():
        close(v.numpy(), P.params("after_one_step")[k])


@needs_mnist
def test_a_batch_must_be_one_block():
    ns = namespace()
    ns["cut"] = lambda img: torch.tensor(P.cut(img.numpy())[0])
    with pytest.raises(RuntimeError) as error:
        run_script(("pad-cut",), ns)
    assert str(error.value) == P.STACK_ERROR
    assert [list(s.shape) for s in ns["small"]] == P.crops()["shapes"]
    run_script(("pad-to-block",), ns)
    assert list(ns["batch"].shape) == P.crops()["block_shape"]
    close(ns["batch"].numpy(), P.crops()["block"])


@needs_text
def test_lines_of_text(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text(P.corpus())
    ns = namespace()
    from pathlib import Path
    ns.update(Path=Path, TEXT=str(path))
    run_script(("pad-text", "pad-sequences"), ns)
    assert ns["vocab"] == P.vocab() and ns["lines"] == P.lines()
    b = P.lines_batch()
    assert np.array_equal(ns["batch"].numpy(), b["batch"])
    assert np.array_equal(ns["mask"].numpy(), b["mask"])
    torch.manual_seed(0)
    run_script(("pad-masked-mean",), ns)
    ratio = (ns["plain"] / ns["mean"]).detach().numpy()
    np.testing.assert_allclose(ratio, np.repeat(np.array(P.masked_mean()["ratio"])[:, None],
                                                16, 1), rtol=1e-5)
    printed(("pad-attention",), ns)
    w = ns["w"].detach().numpy()
    assert np.all(w[~np.broadcast_to(b["mask"][:, None, :], w.shape)] == 0)
    np.testing.assert_allclose(w.sum(-1), 1, rtol=1e-5)
    run_script(("pad-loss",), ns)
    assert ns["loss_fn"].ignore_index == 0
    run_script(("pad-sort",), ns)
    assert ns["lines"] == sorted(P.lines(), key=len)

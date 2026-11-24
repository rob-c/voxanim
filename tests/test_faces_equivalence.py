"""examples/faces.py's own lines, run in PyTorch, give voxanim.study_faces's numbers: the area
resizing, the targets, one Adam step of the detector (the loss with its weighted face cells)
and of the encoder (the triplet loss), the decoding, and torchvision's NMS and IoU. The script's
random scenes and triplets are replaced by the twin's."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import needs_faces
from test_mnist_blocks_equivalence import close, run_script

from voxanim import study_faces as D

torch = pytest.importorskip("torch", reason="torch is not installed: the faces script is "
                                            "checked with it")
tv = pytest.importorskip("torchvision", reason="torchvision is not installed")
F = torch.nn.functional


def namespace():
    from torchvision.ops import box_convert, box_iou, nms
    return dict(torch=torch, nn=torch.nn, F=F, math=math, GRID=D.GRID, CELL=D.CELL,
                FMT=("cxcywh", "xyxy"), box_convert=box_convert, box_iou=box_iou, nms=nms)


def models():
    ns = run_script(("fc-block", "fc-detector", "fc-encoder"), namespace())
    det, enc = ns["detector"], ns["encoder"]
    assert tuple(det.state_dict()) == D.DET_KEYS and tuple(enc.state_dict()) == D.REC_KEYS
    det.load_state_dict({k: torch.tensor(v) for k, v in D.det_init().items()})
    enc.load_state_dict({k: torch.tensor(v) for k, v in D.rec_init().items()})
    assert sum(p.numel() for p in det.parameters()) == D.count(D.det_init()) == 6053
    assert sum(p.numel() for p in enc.parameters()) == D.count(D.rec_init()) == 22304
    return ns, det, enc


@needs_faces
def test_resize_and_targets():
    ns = namespace()
    img = D.photos()[0][7]
    for s in (16, 21, 32):
        ns.update(img=torch.tensor(img), s=s, canvas=torch.zeros(1, 96, 96), x=3, y=5)
        run_script(("fc-paste",), ns)
        close(ns["face"][0, 0].numpy(), D.resize(img, s))
        assert ns["box"] == (3 + s / 2, 5 + s / 2, s, s)
    run_script(("fc-offsets", "fc-targets"), ns)
    sc = D.det_batch(0)["scenes"][0]
    has, want = ns["targets"](sc["boxes"])
    h2, w2 = D.targets(sc["boxes"])
    assert np.array_equal(has.numpy(), h2)
    close(want.numpy(), w2)


@needs_faces
def test_one_detector_step_matches_the_twin():
    ns, det, _enc = models()
    b = D.det_batch(0)
    ns.update(PW=torch.tensor(10.0))
    run_script(("fc-weight",), ns)
    exec("def detector_loss(out, has, want, pw):\n"
         + "\n".join("    " + s for s in __import__("voxanim").blocks.code_lines("fc-loss"))
         + "\n    return loss", ns)
    ns.update(detector=det, x=torch.tensor(b["x"]), has=torch.tensor(b["has"]),
              want=torch.tensor(b["want"]), pw=ns["PW"],
              optimizer=torch.optim.Adam(det.parameters()))
    run_script(("fc-train",), ns)
    assert abs(ns["loss"].item() - D.det_run()["loss"][0]) < 1e-5
    for k, p in det.named_parameters():
        close(p.grad.numpy(), D.det_run()["G0__" + k])
    for k, v in det.state_dict().items():
        close(v.numpy(), D.det_run()["W1__" + k])


@needs_faces
def test_decode_nms_and_iou_match_torchvision():
    ns, det, _enc = models()
    det.load_state_dict({k: torch.tensor(v) for k, v in D.det_params().items()})
    for sc in D.test_scenes()[:12]:
        with torch.no_grad():
            out = det(torch.tensor(sc["canvas"])[None, None])[0]
        ns.update(out=out, keep=0.05)
        run_script(("fc-decode",), ns)
        boxes, cells = D.decode(D.det_forward(D.det_params(), sc["canvas"][None, None])[0],
                                0.05)[0::2]
        np.testing.assert_allclose(ns["boxes"].numpy(), boxes, rtol=1e-6, atol=1e-5)
        score = out[0].sigmoid()[ns["r"], ns["c"]]
        ns.update(score=score)
        run_script(("fc-nms",), ns)
        f = D.find(D.det_params(), sc["canvas"], keep=0.05)
        np.testing.assert_allclose(ns["found"].numpy(), f["found"], rtol=1e-6, atol=1e-5)
        ns.update(truth=sc["boxes"])
        run_script(("fc-iou",), ns)
        close(tv.ops.box_iou(ns["found"], ns["real"]).numpy(), D.iou(f["found"], D.corners(
            sc["boxes"])))
        assert len(cells) == len(boxes)


@needs_faces
def test_one_encoder_step_matches_the_twin():
    ns, _det, enc = models()
    i, j, k = D.triplets(0)
    xs = torch.tensor(D.small())
    ns.update(encoder=enc, anchor=xs[i], same=xs[j], other=xs[k])
    run_script(("fc-triplet",), ns)
    assert abs(ns["loss"].item() - D.rec_run()["loss"][0]) < 1e-5
    opt = torch.optim.Adam(enc.parameters())
    opt.zero_grad()
    ns["loss"].backward()
    for key, p in enc.named_parameters():
        close(p.grad.numpy(), D.rec_run()["G0__" + key])
    opt.step()
    for key, v in enc.state_dict().items():
        # Adam's first step moves every weight by about lr whatever its gradient's size, so
        # where the gradient is next to nothing, rounding decides; compare the rest
        g = D.rec_run()["G0__" + key]
        real = np.abs(g) > 1e-6
        close(v.numpy()[real], D.rec_run()["W1__" + key][real])
        assert np.abs(v.numpy() - D.rec_run()["W1__" + key]).max() <= 2e-3


@needs_faces
def test_verification_and_identification_lines():
    _ns, _det, enc = models()
    enc.load_state_dict({k: torch.tensor(v) for k, v in D.rec_params().items()})
    person = torch.tensor(D.photos()[1])
    small = torch.tensor(D.small())
    ns = dict(namespace(), encoder=enc, small=small, person=person, test=person >= 30,
              THRESHOLD=D.recognition()["threshold"])
    with torch.no_grad():
        run_script(("fc-verify", "fc-identify", "fc-strangers"), ns)
    r = D.recognition()
    assert int(ns["yes"].sum()) == r["true_accept"] + r["false_accept"]
    assert int((ns["yes"] == ns["same"]).sum()) == r["right"]
    who = np.array(D.ENROLLED)[ns["who"].numpy()]
    assert int((who == np.repeat(D.ENROLLED, 9)).sum()) == r["identified"]
    assert int((ns["d"] < ns["THRESHOLD"]).sum()) == r["let_in"]

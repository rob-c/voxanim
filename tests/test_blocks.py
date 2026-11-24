"""The series' shared module (voxanim.blocks): the code regions are read exactly from the
example programs, the formatting helpers, the alphabet's colours in both themes, and the
furniture every animation of the series draws (heading, code panel, loop map, tag)."""

from __future__ import annotations

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, theme
from voxanim.blocks import Blocks, Code

SCRIPT_REGIONS = {"imports", "device", "data", "one-digit", "loader", "one-batch", "model",
                  "print-model", "loss", "optimizer", "train-mode", "train", "step", "grad",
                  "evaluate", "save", "load"}
PARTS_REGIONS = {"parts-imports", "count", "conv", "pool", "dropout", "batchnorm", "layernorm",
                 "embedding", "attention", "softmax", "dropout-model", "mistake-no-zero-grad",
                 "mistake-no-eval", "mistake-double-softmax", "mistake-no-flatten",
                 "cnn-mistake-in-features", "cnn-mistake-channels", "cnn-mistake-no-batch",
                 "cnn-fix-no-batch"}
CNN_REGIONS = {"cnn-" + n for n in SCRIPT_REGIONS}


# ---------------------------------------------------------------------- the code regions
def test_every_region_is_there_once():
    r = B.regions()
    assert set(r) == SCRIPT_REGIONS | PARTS_REGIONS | CNN_REGIONS
    assert {n for n, (p, _l) in r.items() if p == B.SCRIPT} == SCRIPT_REGIONS
    assert {n for n, (p, _l) in r.items() if p == B.CNN_SCRIPT} == CNN_REGIONS
    assert B.code_file("model") == "examples/mnist_blocks.py"
    assert B.code_file("conv") == "examples/blocks_parts.py"


def test_regions_are_read_exactly():
    assert B.code_lines("model") == (
        "model = nn.Sequential(",
        "    nn.Flatten(),",
        "    nn.Linear(784, 64),",
        "    nn.ReLU(),",
        "    nn.Linear(64, 32),",
        "    nn.ReLU(),",
        "    nn.Linear(32, 10),",
        ").to(device)",
    )
    # a nested region, dedented; the outer one keeps its indentation and skips the markers
    assert B.code_lines("step") == (
        "x, y = x.to(device), y.to(device)",
        "logits = model(x)",
        "loss = loss_fn(logits, y)",
        "optimizer.zero_grad()",
        "loss.backward()",
        "optimizer.step()",
    )
    train = B.code_lines("train")
    assert train[:2] == ("for epoch in range(1):", "    for x, y in loader:")
    assert train[2:8] == tuple("        " + s for s in B.code_lines("step"))
    assert not any("region" in s for s in train)
    assert B.code_lines("one-digit") == ("x, y = train_set[0]", "print(x.shape, y)")
    assert B.code_lines("print-model") == ("print(model)",)
    assert B.code_lines("train-mode") == ("model.train()",)
    assert B.code_lines("grad") == ("print(model[1].weight.grad.shape)",)
    assert len(train) == 9          # the panel shows train[:8]: the loop and its step
    assert B.code_line("loader", "batch_size") == "                    batch_size=64,"
    with pytest.raises(ValueError):
        B.code_line("model", "nn.ReLU()")             # two lines hold it
    with pytest.raises(KeyError):
        B.code_lines("no-such-region")


def test_comments_are_left_out_but_kept_on_request():
    with_comments = B.code_lines("model", comments=True)
    assert any("#" in s for s in with_comments)
    assert not any("#" in s for s in B.code_lines("model"))
    text = "\n".join(B.SCRIPT.read_text().splitlines())
    for s in B.code_lines("data"):
        assert s.strip() in text                      # every line is the file's, verbatim


def test_a_hash_inside_a_string_is_not_a_comment(tmp_path):
    f = tmp_path / "x.py"
    f.write_text('# region: a\n    s = "#1"  # the first\n    t = 2\n# endregion\n')
    lines = B._regions(f)["a"]
    assert B._uncomment(lines) == ('s = "#1"', "t = 2")
    f.write_text("# region: a\nx = 1\n")
    with pytest.raises(ValueError):
        B._regions(f)


def test_every_region_on_screen_fits_the_panel():
    for name in B.regions():
        if name in ("imports", "cnn-imports"):
            continue
        for s in B.code_lines(name):
            assert len(s) <= B.MAX_CHARS, (name, s)
    width = layout.text_width("x" * B.MAX_CHARS, B.CODE, "mono")
    x0, x1, _top, _bottom = B.PANEL
    assert width + 20 / 960 <= x1 - x0


def test_the_whole_program_in_order():
    p = B.program()
    assert p[0] == "import sys" and p[-1] == "model.load_state_dict(state)"
    assert not any("region" in s or "#" in s for s in p)
    joined = "\n".join(p)
    pos = [joined.index("\n".join(B.code_lines(n)).split("\n")[0]) for n in
           ("device", "data", "loader", "model", "loss", "optimizer", "train", "evaluate", "save")]
    assert pos == sorted(pos)
    assert 40 <= len(p) <= 70


# ---------------------------------------------------------------------- words and numbers
def test_formatting():
    assert B.shape((64, 1, 28, 28)) == "[64, 1, 28, 28]"
    assert B.torch_size([1, 28, 28]) == "torch.Size([1, 28, 28])"
    assert B.val(-0.4242) == "−0.42" and B.val(2.8215) == "2.82" and B.val(-0.001) == "0.00"
    assert B.commas(50176) == "50,176"
    assert B.words("a tensor: numbers in a block, with a shape") == 9
    assert B.words("its shape: [1, 28, 28]") == 2
    assert B.is_evidence(B.exact("x")) and B.is_evidence(B.measured("y"))
    assert not B.is_evidence("exactly")


def test_samples_and_shades():
    a = np.arange(784 * 64).reshape(784, 64)
    corner, cut_r, cut_c = B.sample(a, 12, 10)
    assert corner.shape == (12, 10) and cut_r and cut_c
    assert B.sample(a[:3, :3], 12, 10)[1:] == (False, False)
    rgb = B.shades(np.array([[0.0, 0.5, 1.0]]), B.DATA, 0, 1)
    light = rgb[0].sum(1)
    assert light[0] > light[1] > light[2]             # deeper: bigger
    grey = B.shades(np.array([[0.0, 1.0]]), B.DATA, 0, 1, zero=np.array([[False, True]]))
    np.testing.assert_allclose(grey[0, 1], B.ZERO)


def test_the_alphabet_is_distinct_in_both_themes():
    inks = [B.DATA, B.WEIGHT, B.GRAD, B.LOSS, B.OPTIM, B.RATE]
    for convert in (lambda c: np.asarray(c, np.float32), theme.to_dark):
        cols = [np.asarray(convert(np.asarray(c, np.float32))) for c in inks]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                assert np.abs(cols[i] - cols[j]).max() > 0.2
    for c in inks:                                    # vivid: dark mode leaves them alone
        np.testing.assert_allclose(theme.to_dark(np.asarray(c, np.float32)), c, atol=0.02)
    assert set(B.STATION_INK) == set(B.STATIONS)


# ---------------------------------------------------------------------- the furniture
class Toy(Blocks):
    """The smallest member of the series: two phases, no study."""

    TITLE = "Toy"
    PHASES = (("one", 24.0), ("two", 24.0))
    LAST = "something before"
    NEXT = "something after"
    STATION = ("data", "loss")
    EVIDENCE = (B.exact("one"), B.measured("two"))
    HEADINGS = ("first heading", ((0.0, "second heading"), (0.5, "third heading")))
    CODE = (Code("step", run=((0.0, 0), (0.5, 2)), out=((0.6, "torch.Size([64, 10])"),)),
            Code("step", run=((0.0, 4),)))

    def phase_1(self, sc, u, op):
        self.box(sc, 0.2, 0.6, 40, 40, B.DATA, op, text="a")

    def phase_2(self, sc, u, op):
        self.box(sc, 0.2, 0.6, 40, 40, B.LOSS, op, text="b")

    def subtitle_for(self, k):
        return "a subtitle"


def visible(a, t, kind=None):
    return [x.s for x in a.scene(t).texts if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def test_the_heading_says_last_time_then_each_phase_then_next():
    a = Toy()
    assert a.heading_at(1.0) == "last time: something before"
    assert a.heading_at(10.0) == "first heading"
    assert a.heading_at(26.0) == "second heading" and a.heading_at(40.0) == "third heading"
    assert a.heading_at(a.t_done - 1.0) == "next: something after"
    assert a.heading_at(a.duration - 0.5) == "next: something after"
    assert "last time: something before" in visible(a, 2.0)
    assert "next: something after" in visible(a, a.t_done - 1.0)


def test_the_code_panel_shows_the_region_and_the_running_line():
    a = Toy()
    code = visible(a, 6.0, "mono")
    assert list(B.code_lines("step")) == [s for s in code if not s.startswith("→")]
    assert a._line_y(6.0, B.code_lines("step")) == pytest.approx(B.CODE_Y)
    assert a._line_y(13.0, B.code_lines("step")) == pytest.approx(B.CODE_Y + 2 * B.CODE_STEP)
    mid = a._line_y(12.2, B.code_lines("step"))      # sliding from line 0 to line 2
    assert B.CODE_Y < mid < B.CODE_Y + 2 * B.CODE_STEP
    assert "→ torch.Size([64, 10])" in visible(a, 16.0, "mono")
    assert "→ torch.Size([64, 10])" not in visible(a, 10.0, "mono")
    assert "examples/mnist_blocks.py" in visible(a, 10.0)
    # the same lines in the next phase: the panel does not blink at the boundary
    texts = [x for x in a.scene(24.0).texts if x.kind == "mono"]
    assert texts and min(x.opacity for x in texts if x.s in B.code_lines("step")) > 0.5


def test_the_loop_map_lights_the_station_and_the_tag_says_what_it_is():
    a = Toy()
    for t, kind in ((10.0, "exact arithmetic: one"), (34.0, "measured on MNIST: two")):
        shown = visible(a, t)
        assert set(B.STATIONS) <= set(shown) and "you are here" in shown
        assert kind in shown
    assert a._tl_station.at(10.0) == "data" and a._tl_station.at(34.0) == "loss"


def test_the_furniture_stays_and_the_loop_closes():
    a = Toy()

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    b = a.spans[0][1]                             # between the phases: the content dips ...
    at = a.scene(b)
    shown = [x.s for x in at.texts if x.opacity > 0.5]
    assert {"data", "loss", "loss = loss_fn(logits, y)"} <= set(shown)   # ... not this
    assert len(at.quads) >= 7                     # the panel, the box, the five stations


def test_the_panel_can_join_regions_and_pick_lines():
    c = Code(("loss", "step"), pick=(0, 3), run=((0.0, 1),))
    assert c.names() == ("loss", "step")
    assert c.lines() == ("loss_fn = nn.CrossEntropyLoss()", "loss = loss_fn(logits, y)")

    class Joined(Toy):
        CODE = (c, Code("step"))
    a = Joined()
    shown = visible(a, 10.0, "mono")
    assert shown == list(c.lines())


def test_too_much_code_is_refused():
    class Long(Toy):
        CODE = (Code("train"), Code("step"))

    with pytest.raises(ValueError):
        Long()

    class Wide(Toy):
        CODE = (Code("imports"), Code("step"))

    with pytest.raises(ValueError):
        Wide()


def test_the_alphabet_draws():
    a = Toy()
    from voxanim.scene import Scene
    sc = Scene()
    a._cap = 1.0
    a.card(sc, 0.2, 0.5, 60, 60, 1.0)
    cell = a.tiles(sc, np.arange(12.0).reshape(3, 4), 0.2, 0.5, 40, 30, 1.0, ring=[(1, 1)])
    assert cell(0, 0)[0] < cell(0, 3)[0] and cell(0, 0)[1] < cell(2, 0)[1]
    a.strip(sc, np.arange(784.0), 0.3, 0.8, 300, 14, 1.0)
    a.bias_strip(sc, np.arange(64.0), 0.3, 0.7, 200, 1.0)
    a.grad_tiles(sc, np.ones((3, 3)), 0.2, 0.5, 30, 30, 1.0)
    a.meter(sc, 0.4, 0.6, 0.5, 1.0, label="2.30")
    a.hand(sc, 0.4, 0.4, 1.0)
    a.dial(sc, 0.3, 0.4, 0.3, 1.0, label="lr 0.1")
    a.stack(sc, 0.2, 0.5, 40, 40, 5, 1.0)
    a.dims(sc, (0.1, 0.3), (0.3, 0.3), "28", 1.0)
    a.big(sc, 0.3, 0.6, "784", 1.0, label="numbers")
    cell = a.digit_card(sc, np.zeros((28, 28)), 0.2, 0.5, 5.0, 1.0)
    assert cell(0, 0) == a.digit_cells(0.2, 0.5, 5.0)(0, 0)
    x0, y0, x1, y1 = a.outline(sc, cell, 3, 4, 3, 3, 5.0, 1.0)
    assert x0 < cell(3, 4)[0] < cell(5, 6)[0] < x1 and y0 < cell(3, 4)[1] < y1
    assert abs(B.DATA_LO + 0.4242) < 1e-4 and abs(B.DATA_HI - 2.8215) < 1e-4
    a.running(sc, 0.3, 0.7, [(0.0, "1"), (0.5, "64")], 0.6, 1.0)
    assert len(sc.quads) > 20 and "64" in [x.s for x in sc.texts]
    assert "784" in [x.s for x in sc.texts] and "lr 0.1" in [x.s for x in sc.texts]


def test_the_cnn_program_is_the_course_program_with_one_block_changed():
    """examples/mnist_cnn.py: the same lines as mnist_blocks.py in every region but the
    model (and the gradient line, which names the first block that learns, and the file
    the weights are saved to)."""
    differ = {"model", "print-model", "grad", "save", "load"}
    for name in SCRIPT_REGIONS - differ:
        assert B.code_lines("cnn-" + name) == B.code_lines(name), name
    assert B.code_lines("cnn-model") == (
        "model = nn.Sequential(",
        "    nn.Conv2d(1, 8, 3), nn.ReLU(),",
        "    nn.MaxPool2d(2),",
        "    nn.Conv2d(8, 16, 3), nn.ReLU(),",
        "    nn.MaxPool2d(2),",
        "    nn.Flatten(),",
        "    nn.Linear(16 * 5 * 5, 10),",
        ").to(device)",
    )
    assert B.code_lines("cnn-grad") == ("print(model[0].weight.grad.shape)",)
    assert B.code_file("cnn-model") == "examples/mnist_cnn.py"
    p, q = B.program(B.SCRIPT), B.program(B.CNN_SCRIPT)
    assert len(p) == len(q) and sum(a != b for a, b in zip(p, q, strict=True)) == 9

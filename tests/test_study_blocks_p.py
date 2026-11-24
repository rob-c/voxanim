"""The numbers behind animations 230-231 (voxanim.study_blocks_p): four digits cut to their ink,
torch.stack's error, and the padded block [4, 1, 28, 28]; eight lines of Tiny Shakespeare
padded with id 0 into [8, 50], and their mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import needs_mnist, needs_text

from voxanim import blocks as B
from voxanim import study_blocks_p as S
from voxanim import study_course as course
from voxanim import study_padding as P

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue_titles():
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("number"), int) and "title" in o:
                found[o["number"]] = o["title"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(CATALOGUE.read_text()))
    return found


# ---------------------------------------------------------------------- no data needed
def test_the_titles_are_the_catalogues():
    titles = catalogue_titles()
    assert (titles[230], titles[231]) == ("A batch is one block", "Padding lines of text")
    assert titles[229] == "The CNN with padding, measured"
    assert titles[232] == "Masks: ignore the padding"


def test_a_printed_list_is_wrapped_without_losing_anything():
    shapes = [[1, 20, 20], [1, 20, 17], [1, 20, 20], [1, 20, 14]]
    lines = S.wrap_list(shapes)
    assert lines == ["[[1, 20, 20], [1, 20, 17],", "[1, 20, 20], [1, 20, 14]]"]
    assert " ".join(lines) == str(shapes)
    assert S.wrap_list([[1, 2]]) == ["[[1, 2]]"]
    assert all(len(s) <= S.OUT_WIDTH for s in S.wrap_list([[1, 20, 20]] * 9))


def test_the_code_the_two_show():
    assert B.code_lines("pad-cut")[-1] == "batch = torch.stack(small)"
    assert B.code_line("pad-to-block", "F.pad") == "    return F.pad(img, room, value=BG)"
    assert B.code_lines("pad-sequences")[-1] == "mask = batch != 0"
    assert B.code_file("pad-text") == "examples/padding.py"


# ---------------------------------------------------------------------- 230
@needs_mnist
def test_a_batch_is_a_block():
    s = S.a_batch_is_a_block().validate()
    m = s.metadata
    assert m["labels"] == [5, 0, 4, 1]
    assert m["sizes"] == ["20 × 20", "20 × 17", "20 × 20", "20 × 14"]
    assert m["shape_texts"] == ["[1, 20, 20]", "[1, 20, 17]", "[1, 20, 20]", "[1, 20, 14]"]
    assert m["printed_shapes"] == "[[1, 20, 20], [1, 20, 17], [1, 20, 20], [1, 20, 14]]"
    # torch.stack's own error, wrapped for the output slot, nothing lost
    assert m["error"] == "RuntimeError: " + P.STACK_ERROR
    assert " ".join(m["error_lines"]) == m["error"]
    assert all(len(line) <= S.OUT_WIDTH for line in m["error_lines"])
    assert m["other"] == 1 and "[1, 20, 17] at entry 1" in m["error"]
    # the paper's value
    assert m["bg_text"] == "−0.42" and m["bg_code"] == "BG = -0.4242"
    assert abs(m["bg"] - (0 - course.MEAN) / course.STD) < 1e-6
    # the padding, counted
    assert m["pads"] == [384, 444, 384, 504] and m["real"] == [400, 340, 400, 280]
    assert (m["padding"], m["cells"], m["real_total"]) == (1_716, 3_136, 1_420)
    assert m["running_pads"] == [384, 828, 1_212, 1_716]
    assert m["running_cells"] == [784, 1_568, 2_352, 3_136]
    assert round(100 * m["padding_share"]) == 55
    assert m["room"][1] == [0, 11, 0, 8]                  # left, right, top, bottom
    assert m["block_text"] == "[4, 1, 28, 28]"
    assert m["printed_block"] == "padded: torch.Size([4, 1, 28, 28])"
    # the pictures: the real digits, and the padded block made of the pieces and BG
    full = s.panels[0].data["values"]
    block = s.panels[2].data["values"]
    for i, ((r0, r1, c0, c1), (_c, h, w)) in enumerate(zip(m["boxes"], m["shapes"])):
        np.testing.assert_array_equal(full[i], course.digit(i)[0][0])
        np.testing.assert_array_equal(block[i, :h, :w], full[i][r0:r1 + 1, c0:c1 + 1])
        pad = np.ones((28, 28), bool)
        pad[:h, :w] = False
        assert np.allclose(block[i][pad], P.BG) and int(pad.sum()) == m["pads"][i]
    assert s.panels[3].data["values"].sum() == 1_716


# ---------------------------------------------------------------------- 231
@needs_text
def test_lines_of_text():
    s = S.lines_of_text().validate()
    m = s.metadata
    assert m["text"][:2] == ["First Citizen:", "Before we proceed any further, hear me speak."]
    assert m["lengths"] == [14, 45, 4, 13, 14, 50, 4, 19] == [len(t) for t in m["text"]]
    assert (m["longest"], m["longest_row"]) == (50, 5)
    assert (m["real"], m["cells"], m["padding"]) == (163, 400, 237)
    assert round(100 * m["padding_share"]) == 59
    assert m["running_real"][-1] == 163 and m["running_real"][:2] == [14, 59]
    assert m["vocab_size"] == 66 and m["top_id"] == 65
    assert m["first"] == [["F", 19], ["i", 48], ["r", 57]]
    assert m["batch_text"] == "[8, 50]" and m["printed_shape"] == "torch.Size([8, 50])"
    assert m["printed"] == "lines: torch.Size([8, 50]) 163 real of 400"
    assert " ".join(m["printed_lines"]) == m["printed"]
    assert m["seq_out"] == "seqs[0] = tensor([19, 48, 57, ...])"
    for out in (m["lengths_out"], m["vocab_out"], m["seq_out"], *m["printed_lines"]):
        assert len("→ " + out) <= B.MAX_CHARS
    batch = s.panels[1].data["values"]
    mask = s.panels[2].data["values"]
    assert batch.shape == (8, 50)
    np.testing.assert_array_equal(mask, (batch != 0).astype(float))
    np.testing.assert_array_equal(batch[0, :14], P.encode("First Citizen:"))
    assert (batch[0, 14:] == 0).all() and (batch[5] != 0).all()
    assert mask.sum(1).tolist() == m["lengths"]

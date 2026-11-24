"""The numbers behind animations 230-231 of "Blocks and code: padding" (group ``p``): padding as
the way to make one batch, one block of numbers, out of things that differ in size
(``examples/padding.py``, :mod:`voxanim.study_padding`).

* ``padding-a-batch-is-a-block``: the first four training digits (``x[i]`` of the script)
  cut to their ink are 20 x 20, 20 x 17, 20 x 20 and 20 x 14 (:func:`voxanim.study_padding.
  crops`), so ``torch.stack`` stops with PyTorch's own error
  (:data:`voxanim.study_padding.STACK_ERROR`); padded on the right and at the bottom with
  the paper's value (:data:`voxanim.study_padding.BG`, -0.42) to 28 x 28 they stack into one
  block, ``[4, 1, 28, 28]``, of which 1,716 of 3,136 numbers are padding.
* ``padding-lines-of-text``: the first eight lines of Tiny Shakespeare, 4 to 50 characters,
  each character swapped for its id, padded with id 0 to the longest
  (``pad_sequence(batch_first=True)``): an ``[8, 50]`` block of which 163 of 400 cells are
  real, and the mask ``batch != 0`` that says which (59% padding).

Everything is exact arithmetic on the real digits and the real text, checked here against
the script's rules; ``tests/test_padding_equivalence.py`` checks the same numbers in torch.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from . import study_padding as P
from .blocks import code_line, shape, torch_size
from .study import Panel, Study
from .study_blocks_m import wrap_error

SIDE = P.SIDE                  # pad_to's side: 28
LINES = P.TEXT_LINES           # the eight lines of region pad-sequences
SHOWN_IDS = 3                  # the first characters of the first line shown with their id
OUT_WIDTH = 39                 # characters of an output-slot line ("→ " first)
SOURCE_DIGITS = ("the first four MNIST training digits, normalised as the script's "
                 "Compose([ToTensor(), Normalize(...)]) leaves them (voxanim.study_course)")
SOURCE_TEXT = "the first eight lines of Tiny Shakespeare that are not empty (voxanim.realdata)"


def wrap_list(items, width=OUT_WIDTH):
    """``print(items)`` of a list of shapes, broken after a comma between two items into as
    few lines of at most ``width`` characters as it takes, the items shared out evenly: the
    lines joined by a space are the printed text."""
    parts = [str(list(s)) for s in items]
    n = len(parts)
    for count in range(1, n + 1):
        per = -(-n // count)
        lines = [", ".join(parts[i:i + per]) for i in range(0, n, per)]
        lines = [("[" if i == 0 else "") + s + ("]" if i == len(lines) - 1 else ",")
                 for i, s in enumerate(lines)]
        if max(len(s) for s in lines) <= width:
            break
    else:
        raise ValueError("an item is wider than %d characters" % width)
    if " ".join(lines) != str([list(s) for s in items]):
        raise ValueError("the wrapped list is not the printed list")
    return lines


# ====================================================================== 230
def a_batch_is_a_block():
    c = P.crops()
    pieces, boxes = c["pieces"], c["boxes"]
    digits = [course.digit(i) for i in range(P.CUT_DIGITS)]
    full = [np.asarray(img[0], float) for img, _y in digits]
    labels = [int(y) for _img, y in digits]
    shapes = [list(p.shape) for p in pieces]
    # cut: the box holds every pixel brighter than the paper, and nothing but paper is left out
    for img, piece, (r0, r1, c0, c1) in zip(full, pieces, boxes, strict=True):
        if not np.array_equal(img[r0:r1 + 1, c0:c1 + 1], np.asarray(piece[0], float)):
            raise ValueError("a cut is not its box of the digit")
        outside = img.copy()
        outside[r0:r1 + 1, c0:c1 + 1] = img.min()
        if not np.allclose(outside, img.min()):
            raise ValueError("ink was left outside a cut")
    if not np.isclose(min(float(f.min()) for f in full), P.BG, atol=1e-5):
        raise ValueError("the paper is not BG")
    # torch.stack's error names entry 0 and the first entry of another size
    other = next(i for i, s in enumerate(shapes) if s != shapes[0])
    said = "but got %s at entry 0 and %s at entry %d" % (shape(shapes[0]), shape(shapes[other]),
                                                         other)
    if said not in P.STACK_ERROR:
        raise ValueError("torch.stack's error is not about entries 0 and %d" % other)
    error = "RuntimeError: " + P.STACK_ERROR
    # pad_to: the piece in the top-left corner, BG on the right and at the bottom
    block = np.asarray(c["block"], float)
    pads = []
    for piece, padded in zip(pieces, block, strict=True):
        h, w = piece.shape[-2:]
        if not np.array_equal(padded[0, :h, :w], np.asarray(piece[0], float)):
            raise ValueError("a padded digit does not start with its piece")
        mask = np.ones((SIDE, SIDE), bool)
        mask[:h, :w] = False
        if not np.allclose(padded[0][mask], P.BG):
            raise ValueError("the padding is not BG")
        pads.append(int(mask.sum()))
    room = [[0, SIDE - s[2], 0, SIDE - s[1]] for s in shapes]      # left, right, top, bottom
    real = [s[1] * s[2] for s in shapes]
    if sum(pads) != c["padding"] or sum(real) != c["real"] or c["cells"] != sum(pads) + sum(real):
        raise ValueError("the padding does not add up")
    if c["block_shape"] != [P.CUT_DIGITS, 1, SIDE, SIDE]:
        raise ValueError("the padded block is not [4, 1, 28, 28]")
    running = np.cumsum(pads).tolist()
    meta = dict(
        source=SOURCE_DIGITS, evidence="exact arithmetic on the real digits, and torch's own "
        "error", labels=labels, boxes=[list(b) for b in boxes], shapes=shapes,
        shape_texts=[shape(s) for s in shapes],
        sizes=["%d × %d" % (s[1], s[2]) for s in shapes],
        printed_shapes=str(shapes), printed_lines=wrap_list(shapes),
        error=error, error_lines=wrap_error(error, OUT_WIDTH), other=other,
        bg=float(P.BG), bg_text="−%.2f" % abs(P.BG), bg_code="BG = %.4f" % P.BG,
        side=SIDE, room=room, real=real, pads=pads, running_pads=[int(v) for v in running],
        running_cells=[SIDE * SIDE * (i + 1) for i in range(len(pads))],
        padding=int(c["padding"]), cells=int(c["cells"]), real_total=int(c["real"]),
        padding_share=c["padding"] / c["cells"],
        block_shape=c["block_shape"], block_text=shape(c["block_shape"]),
        printed_block="padded: " + torch_size(c["block_shape"]),
        cut_code=code_line("pad-cut", "cut(x[i])"), stack_code=code_line("pad-cut", "stack"),
        pad_code=code_line("pad-to-block", "F.pad"),
        limitations=["the digits are cut by the script's cut(): the smallest box holding every "
                     "pixel above the paper's value; the paper the cut leaves out is all "
                     "−0.42",
                     "the padding has the paper's value, so it looks like paper; it is marked "
                     "with grey stripes only so that it can be seen",
                     "the error is PyTorch's (checked by tests/test_padding_equivalence.py); "
                     "the script catches it with try/except, left out of region pad-cut"])
    panels = (
        Panel("Four digits", "x[0] to x[3], the first four training digits, normalised",
              "images", dict(values=np.stack(full), labels=[str(y) for y in labels])),
        Panel("Cut to their ink", "numbers in each digit cut to its ink (rows x columns)",
              "bars", dict(values=np.array(real, float), labels=meta["sizes"])),
        Panel("Padded to 28 x 28", "each piece with the paper's value (−0.42) on the right "
              "and at the bottom: one block [4, 1, 28, 28]", "images",
              dict(values=block[:, 0], labels=[str(y) for y in labels])),
        Panel("How much is padding", "padding numbers per digit; %s of %s in all"
              % (format(c["padding"], ","), format(c["cells"], ",")), "bars",
              dict(values=np.array(pads, float), labels=meta["sizes"])),
    )
    return Study("A batch is one block", "Four digits cut to their ink have four sizes, so "
                 "torch.stack fails; padded to 28 x 28 they make one block", panels, meta)


# ====================================================================== 231
def lines_of_text():
    b = P.lines_batch()
    vocab = P.vocab()
    text, lengths = b["text"], b["lengths"]
    batch, mask = np.asarray(b["batch"]), np.asarray(b["mask"])
    if vocab[0] != "<pad>" or any(ch == "<pad>" for ch in vocab[1:]):
        raise ValueError("id 0 is not kept for padding")
    longest = max(lengths)
    if batch.shape != (LINES, longest):
        raise ValueError("the batch is not [8, longest]")
    for i, s in enumerate(text):
        if not np.array_equal(batch[i, :len(s)], P.encode(s)) or batch[i, len(s):].any():
            raise ValueError("line %d is not its ids then 0s" % i)
        if (batch[i, :len(s)] == 0).any():
            raise ValueError("a real character has id 0")
    if int(mask.sum()) != sum(lengths) or b["real"] != sum(lengths):
        raise ValueError("the mask does not count the real characters")
    cells = int(batch.size)
    pads = [longest - n for n in lengths]
    first = text[0][:SHOWN_IDS]
    ids = [int(v) for v in batch[0, :SHOWN_IDS]]
    meta = dict(
        source=SOURCE_TEXT, evidence="exact arithmetic on the real text",
        text=list(text), lengths=list(lengths), longest=longest, longest_row=int(np.argmax(
            lengths)), vocab_size=len(vocab), top_id=len(vocab) - 1,
        first=[[ch, i] for ch, i in zip(first, ids, strict=True)],
        first_ids=[int(v) for v in batch[0, :lengths[0]]],
        pads=pads, padding=int(sum(pads)), real=int(b["real"]), cells=cells,
        padding_share=float(b["padding_share"]),
        running_real=[int(v) for v in np.cumsum(lengths)],
        batch_shape=list(batch.shape), batch_text=shape(batch.shape),
        printed_shape=torch_size(batch.shape),
        printed="lines: %s %d real of %d" % (torch_size(batch.shape), b["real"], cells),
        printed_lines=["lines: %s" % torch_size(batch.shape),
                       "%d real of %d" % (b["real"], cells)],
        lengths_out="lengths " + ", ".join(str(n) for n in lengths),
        vocab_out='vocab[0] = "<pad>", %d ids' % len(vocab),
        seq_out="seqs[0] = tensor([%s, ...])" % ", ".join(str(i) for i in ids),
        limitations=["the characters are shaded by their id (deeper: a bigger id); the id is "
                     "a name, not an amount",
                     "the pads (id 0) are marked with grey stripes so that they can be seen",
                     "the mask is drawn in place of the batch, row by row"])
    panels = (
        Panel("Eight lines", "characters in each of the first eight lines", "bars",
              dict(values=np.array(lengths, float), labels=[str(i) for i in range(LINES)])),
        Panel("Padded", "pad_sequence(seqs, batch_first=True): the ids, 0 for a pad",
              "matrix", dict(values=batch.astype(float))),
        Panel("The mask", "batch != 0: 1 for a real character, 0 for a pad", "matrix",
              dict(values=mask.astype(float))),
        Panel("Real and padding", "cells of the [8, 50] block: real characters and pads",
              "bars", dict(values=np.array([b["real"], sum(pads)], float),
                           labels=["real", "padding"])),
    )
    return Study("Padding lines of text", "Eight lines of 4 to 50 characters padded with id 0 "
                 "into one [8, 50] block, and the mask of the 163 real cells", panels, meta)


STUDIES = {"padding-a-batch-is-a-block": a_batch_is_a_block,
           "padding-lines-of-text": lines_of_text}

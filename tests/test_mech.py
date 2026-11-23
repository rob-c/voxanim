"""The drawing kit of the "next 100" (voxanim.mech): its number, tick, wrap and colour helpers,
and that every one of the 100 is drawn by it."""

import numpy as np

from voxanim import layout, mech
from voxanim.next_catalog import DRAWN_BY, ITEMS, drawing


def test_every_catalogue_entry_has_its_own_drawing():
    classes = [drawing(item["name"]) for item in ITEMS]
    assert len(ITEMS) == 182 and all(c is not None for c in classes)
    assert all(issubclass(c, mech.Mechanism) for c in classes)
    assert len(set(classes)) == len(ITEMS)                # no two share one generic canvas
    assert sum(last - first + 1 for _m, first, last in DRAWN_BY) == len(ITEMS)
    assert [item["number"] for item in ITEMS] == list(range(1, len(ITEMS) + 1))


def test_numbers_are_short_enough_for_an_axis():
    """sparse-attention-patterns: a y tick of 500,000,000 ran out of the title-safe area."""
    assert mech.number(5e8) == "500M"
    assert mech.number(17_800_000) == "17.8M"
    assert mech.number(130_000) == "130,000"
    assert mech.number(0.4567) == "0.457"
    assert mech.number(0) == "0"
    assert mech.nice_ticks(0, 5e8, 3)[0] == 0
    assert mech.nice_ticks(0.0, 1.0) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_long_text_wraps_to_as_many_lines_as_it_needs():
    lines = mech.fit("These overlapping summaries do not assign causal roles to heads at all.",
                     mech.TEXT, 0.25)
    assert len(lines) >= 3
    assert all(layout.text_width(line, mech.TEXT) <= 0.25 for line in lines)
    assert " ".join(lines).split() == ["These", "overlapping", "summaries", "do", "not", "assign", "causal", "roles", "to", "heads", "at", "all."]  # noqa: E501


def test_heat_is_paper_at_zero_red_up_and_blue_down():
    zero, up, down = mech.heat(0.0), mech.heat(1.0), mech.heat(-1.0)
    assert np.allclose(zero, mech.PAPER)
    assert up[0] > up[2] and down[2] > down[0]
    assert float(np.abs(up - zero).max()) > 0.5


def test_the_colours_are_the_conv2d_family_starting_red_blue_green():
    red, blue, green = mech.INKS[:3]
    assert red[0] > max(red[1], red[2]) and blue[2] > max(blue[0], blue[1]) and green[1] > max(green[0], green[2])  # noqa: E501
    assert len(mech.INKS) == 8

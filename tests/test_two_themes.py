"""Rendering in two themes from the command line: the ``--theme`` option, the ``-dark``
file names, the manifest, and that a dark frame is a different picture with the same
corner mark."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
from conftest import DRAFT

from voxanim import __main__ as cli
from voxanim import brand, presets, theme
from voxanim.scene import Renderer

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _args(*argv):
    return cli.parser().parse_args(list(argv))


def test_the_theme_option():
    assert _args("render", "conv2d").theme is None           # the environment decides
    assert _args("render", "conv2d", "--theme", "dark").theme == "dark"
    assert _args("render", "a", "b", "--theme", "both").theme == "both"
    assert _args("encode", "conv2d", "--theme", "both").theme == "both"
    assert _args("still", "conv2d", "-t", "1", "--theme", "dark").theme == "dark"
    assert _args("sheet", "conv2d", "--theme", "dark").theme == "dark"
    assert _args("previews").theme == "both"
    with pytest.raises(SystemExit):                          # one still is one theme
        _args("still", "conv2d", "-t", "1", "--theme", "both")
    with pytest.raises(SystemExit):
        _args("render", "conv2d", "--theme", "sepia")


def test_dark_files_have_their_own_names(monkeypatch, tmp_path):
    args = _args("render", "dnn-eval", "--digit", "8")
    monkeypatch.setenv("VOXANIM_THEME", "light")
    light = (cli._suffix("dnn-eval", args), cli._frames_dir("dnn-eval", args))
    monkeypatch.setenv("VOXANIM_THEME", "dark")
    dark = (cli._suffix("dnn-eval", args), cli._frames_dir("dnn-eval", args))
    assert light[0] == "_digit8" and dark[0] == "_digit8-dark"
    assert dark[1] == light[1] + "-dark" and light[1].endswith(os.path.join("build", "dnn-eval"
                                                                            "_digit8"))
    assert cli._out_dir(args).endswith(os.path.join("build", "out"))
    # --out moves the encoded files, the manifest and the frames
    args = _args("render", "conv2d", "--out", str(tmp_path))
    assert cli._out_dir(args) == str(tmp_path)
    assert cli._frames_dir("conv2d", args) == str(tmp_path / "frames" / "conv2d-dark")
    still = _args("still", "conv2d", "-t", "1")
    assert cli._out_path(still, "x", "_001.00.png") == os.path.join("x", "conv2d-dark_001.00.png")
    # no media name ends like a dark render or a preview, so the names cannot collide
    for p in presets.REGISTRY.values():
        assert not p.media.endswith(("-dark", "-thumb"))


def test_the_theme_is_set_for_the_run_and_put_back(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_render", lambda args: seen.append(theme.name()))
    monkeypatch.setenv("VOXANIM_THEME", "dark")
    cli.main(["render", "conv2d"])                           # no option: the environment's
    cli.main(["render", "conv2d", "--theme", "light"])
    cli.main(["render", "conv2d", "--theme", "both"])
    assert seen == ["dark", "light", "light", "dark"]
    assert os.environ["VOXANIM_THEME"] == "dark"
    monkeypatch.delenv("VOXANIM_THEME")
    cli.main(["render", "conv2d", "--theme", "dark"])
    assert seen[-1] == "dark" and "VOXANIM_THEME" not in os.environ


def _frame(monkeypatch, dark, mark=True):
    monkeypatch.setenv("VOXANIM_THEME", "dark" if dark else "light")
    monkeypatch.setenv("VOXANIM_BRAND", "1" if mark else "0")
    anim = presets.get("conv2d").build()
    return np.array(Renderer(anim.cam, DRAFT).render_at(anim.scene, 3.3, 1 / 30))


def test_a_dark_frame_differs_from_a_light_one_and_both_carry_the_mark(monkeypatch):
    light, dark = _frame(monkeypatch, False), _frame(monkeypatch, True)
    assert light.shape == dark.shape
    assert light.mean() > 0.6 and dark.mean() < 0.4          # the page turned over
    x0, y0, x1, y1 = brand.box(*light.shape[:2])
    for is_dark, marked in ((False, light), (True, dark)):
        diff = np.abs(marked - _frame(monkeypatch, is_dark, mark=False)).max(-1)
        assert diff[y0:y1, x0:x1].max() > 0.05               # the mark is there
        diff[y0:y1, x0:x1] = 0
        assert diff.max() == 0                               # and nothing else changed


@needs_ffmpeg
def test_render_both_themes_end_to_end(tmp_path, monkeypatch):
    """Two workers, so the dark pass is drawn by processes that must inherit the theme."""
    monkeypatch.delenv("VOXANIM_THEME", raising=False)
    out = tmp_path / "out"
    cli.main(["render", "lesson-stride-1", "--theme", "both", "--quality", "draft",
              "--scale", "0.5", "--fps", "2", "--gif-fps", "2", "--jobs", "2",
              "--out", str(out)])
    media = presets.get("lesson-stride-1").media
    made = sorted(f for f in os.listdir(out) if f != "frames")
    assert made == sorted(media + tail for tail in (
        ".mp4", ".gif", ".json", ".jpg", "-thumb.mp4",
        "-dark.mp4", "-dark.gif", "-dark.json", "-dark.jpg", "-dark-thumb.mp4"))
    assert sorted(os.listdir(out / "frames")) == ["lesson-stride-1", "lesson-stride-1-dark"]
    light = json.loads((out / (media + ".json")).read_text())
    dark = json.loads((out / (media + "-dark.json")).read_text())
    assert (light["theme"], dark["theme"]) == ("light", "dark")
    assert light["style"] == dark["style"]                   # "style" means what it did
    frames = len(os.listdir(out / "frames" / "lesson-stride-1"))
    assert light["frames"] == dark["frames"] == frames > 0
    assert set(light["outputs"]) == {media + ".mp4", media + ".gif"}
    assert set(dark["outputs"]) == {media + "-dark.mp4", media + "-dark.gif"}
    assert light["outputs"][media + ".mp4"] != dark["outputs"][media + "-dark.mp4"]
    assert "VOXANIM_THEME" not in os.environ

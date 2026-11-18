"""The website's previews (voxanim.previews): a poster and a small clip of each video."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from PIL import Image

from voxanim import __main__ as cli
from voxanim import presets, previews

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _video(path, seconds=4):
    """A 960x540 test pattern, the size of a render."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=960x540:rate=30:duration=%d" % seconds,
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_paths_carry_the_theme_of_the_video():
    assert previews.paths("out/x.mp4") == ("out/x.jpg", "out/x-thumb.mp4")
    assert previews.paths("out/x-dark.mp4") == ("out/x-dark.jpg", "out/x-dark-thumb.mp4")


def test_make(tmp_path):
    mp4 = _video(tmp_path / "x.mp4")
    assert previews.stale(mp4) == list(previews.paths(mp4))
    jpg, clip = previews.make(mp4, quiet=True)
    assert (jpg, clip) == previews.paths(mp4) and previews.stale(mp4) == []
    with Image.open(jpg) as im:
        assert im.size == (previews.POSTER_WIDTH, 360)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                            "-of", "csv=p=0", clip], check=True, capture_output=True, text=True)
    assert probe.stdout.strip() == "h264,480,270,10/1"
    silent = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                             "stream=index", "-of", "csv=p=0", clip], capture_output=True,
                            text=True)
    assert silent.stdout.strip() == ""
    # the first half of the 4 s, then the 2 s hold
    assert abs(previews.duration(clip) - 4.0) < 0.2
    assert abs(previews.duration(mp4) - 4.0) < 0.1
    # small: this is all a card downloads
    assert os.path.getsize(jpg) < 60_000 and os.path.getsize(clip) < 60_000
    assert not [f for f in os.listdir(tmp_path) if ".part" in f]


def test_previews_are_kept_until_the_video_is_newer(tmp_path):
    mp4 = _video(tmp_path / "x.mp4", seconds=1)
    jpg, clip = previews.make(mp4, quiet=True)
    os.utime(jpg, (1000, 1000))
    os.utime(clip, (2000, 2000))
    os.utime(mp4, (500, 500))
    previews.make(mp4, quiet=True)
    assert (os.path.getmtime(jpg), os.path.getmtime(clip)) == (1000, 2000)      # kept
    os.utime(mp4, (1500, 1500))                              # rendered again since the poster
    assert previews.stale(mp4) == [jpg]
    previews.make(mp4, quiet=True)
    assert os.path.getmtime(jpg) > 1500 and os.path.getmtime(clip) == 2000
    previews.make(mp4, force=True, quiet=True)
    assert os.path.getmtime(clip) > 2000


def test_the_previews_command_does_both_themes(tmp_path, capsys):
    media = presets.get("conv2d").media
    light = _video(tmp_path / (media + ".mp4"), seconds=1)
    cli.main(["previews", "conv2d", "--out", str(tmp_path), "--jobs", "1"])   # no dark video: fine
    assert previews.stale(light) == []
    assert "no dark MP4" in capsys.readouterr().err
    with pytest.raises(SystemExit):                          # asked for by name, and absent
        cli.main(["previews", "conv2d", "--out", str(tmp_path), "--theme", "dark"])
    dark = _video(tmp_path / (media + "-dark.mp4"), seconds=1)
    cli.main(["previews", "conv2d", "--out", str(tmp_path), "--jobs", "1"])
    assert previews.stale(dark) == []
    assert os.path.exists(tmp_path / (media + "-dark-thumb.mp4"))
    with pytest.raises(SystemExit):                          # an animation with no video
        cli.main(["previews", "bump-hunt", "--out", str(tmp_path), "--theme", "light"])

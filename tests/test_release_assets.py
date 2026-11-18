"""Release packaging (tools/release_assets.py): what a release holds, and what stops one."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import release_assets as release

from voxanim import presets

REGISTRY = {n: SimpleNamespace(name=n, media=m) for n, m in (("one", "first"), ("two", "second"))}


def _render(out, media, theme="light", stamp=1000):
    """A stand-in for one render: a video, a GIF, their manifest and the previews."""
    base = media + release.THEMES[theme]
    outputs = {}
    for ext in ("mp4", "gif"):
        data = ("%s %s" % (base, ext)).encode() * 50
        (out / ("%s.%s" % (base, ext))).write_bytes(data)
        outputs["%s.%s" % (base, ext)] = hashlib.sha256(data).hexdigest()
        os.utime(out / ("%s.%s" % (base, ext)), (stamp, stamp))
    (out / (base + ".json")).write_text(json.dumps(
        {"theme": theme, "fps": 30, "frames": 90, "outputs": outputs}))
    for tail in (".jpg", "-thumb.mp4"):
        (out / (base + tail)).write_bytes(("%s%s" % (base, tail)).encode())
        os.utime(out / (base + tail), (stamp + 10, stamp + 10))


@pytest.fixture
def made(tmp_path, monkeypatch):
    """A tiny build/out with both animations in both themes, and a model cache."""
    monkeypatch.setattr(presets, "REGISTRY", REGISTRY)
    out, cache = tmp_path / "out", tmp_path / "cache"
    out.mkdir()
    cache.mkdir()
    for media in ("first", "second"):
        for theme in release.THEMES:
            _render(out, media, theme)
    (cache / "mlp-0123456789ab.npz").write_bytes(b"weights")
    (cache / "mnist.npz").write_bytes(b"a dataset, not a model")
    return SimpleNamespace(out=out, cache=cache, dest=tmp_path / "release")


def _run(made, *flags, dest=None):
    release.main([str(dest or made.dest), "--renders", str(made.out), "--cache",
                  str(made.cache), *flags])


def _members(path, mode="r:"):
    with tarfile.open(path, mode) as tf:
        return tf.getnames()


def test_a_release_of_both_themes(made, capsys, tmp_path):
    _run(made)
    assert sorted(os.listdir(made.dest)) == sorted([
        "first.mp4", "first-dark.mp4", "second.mp4", "second-dark.mp4",
        "voxanim-gifs-light.tar", "voxanim-gifs-dark.tar", "voxanim-site.tar.gz",
        "voxanim-weights.tar.gz", "SHA256SUMS"])
    # the GIFs are stored, not compressed: "r:" refuses a compressed tar
    assert _members(made.dest / "voxanim-gifs-light.tar") == ["first.gif", "second.gif"]
    assert _members(made.dest / "voxanim-gifs-dark.tar") == ["first-dark.gif", "second-dark.gif"]
    assert _members(made.dest / "voxanim-site.tar.gz", "r:gz") == sorted(
        m + tail for m in ("first", "second")
        for tail in (".jpg", "-thumb.mp4", ".json", "-dark.jpg", "-dark-thumb.mp4", "-dark.json"))
    assert _members(made.dest / "voxanim-weights.tar.gz", "r:gz") == ["mlp-0123456789ab.npz"]
    with tarfile.open(made.dest / "voxanim-gifs-dark.tar") as tf:
        assert tf.extractfile("first-dark.gif").read() == (made.out / "first-dark.gif").read_bytes()
    # SHA256SUMS covers every other file of the release, correctly
    sums = dict(reversed(line.split("  ")) for line in
                (made.dest / "SHA256SUMS").read_text().splitlines())
    assert sorted(sums) == sorted(f for f in os.listdir(made.dest) if f != "SHA256SUMS")
    for name, digest in sums.items():
        assert release.sha256(made.dest / name) == digest
    said = capsys.readouterr().out
    assert "2 animations (2 with a dark render)" in said
    assert "9 files of the 1000 a release may hold" in said
    # the same renders give the same archives, byte for byte
    _run(made, dest=tmp_path / "again")
    again = tmp_path / "again" / "SHA256SUMS"
    assert (made.dest / "SHA256SUMS").read_bytes() == again.read_bytes()


def test_a_missing_dark_render_stops_the_release_unless_light_only(made, capsys):
    for tail in ("-dark.mp4", "-dark.gif", "-dark.json", "-dark.jpg", "-dark-thumb.mp4"):
        os.remove(made.out / ("second" + tail))
    with pytest.raises(SystemExit, match="two: no dark render"):
        _run(made)
    assert not made.dest.exists()                            # nothing half made
    _run(made, "--light-only")
    files = os.listdir(made.dest)
    assert "second.mp4" in files and "second-dark.mp4" not in files and "first-dark.mp4" in files
    assert _members(made.dest / "voxanim-gifs-dark.tar") == ["first-dark.gif"]
    assert "second-dark.json" not in _members(made.dest / "voxanim-site.tar.gz", "r:gz")
    assert "2 animations (1 with a dark render)" in capsys.readouterr().out


def test_light_only_with_no_dark_render_at_all_has_no_dark_archive(made):
    for f in os.listdir(made.out):
        if "-dark" in f:
            os.remove(made.out / f)
    _run(made, "--light-only")
    assert sorted(os.listdir(made.dest)) == sorted([
        "first.mp4", "second.mp4", "voxanim-gifs-light.tar", "voxanim-site.tar.gz",
        "voxanim-weights.tar.gz", "SHA256SUMS"])


def test_an_animation_with_no_render_stops_the_release_unless_partial(made, capsys):
    for f in os.listdir(made.out):
        if f.startswith("second"):
            os.remove(made.out / f)
    with pytest.raises(SystemExit, match="two: no light render"):
        _run(made)
    _run(made, "--allow-partial")
    assert "first.mp4" in os.listdir(made.dest) and "second.mp4" not in os.listdir(made.dest)
    assert "1 animations (1 with a dark render, 1 left out)" in capsys.readouterr().out


def test_a_dark_render_alone_is_not_an_animation(made):
    for tail in (".mp4", ".gif", ".json"):
        os.remove(made.out / ("second" + tail))
    with pytest.raises(SystemExit, match="two: no light render"):
        _run(made, "--allow-partial", "--light-only")


@pytest.mark.parametrize("break_it, says", [
    (lambda out: (out / "first-dark.mp4").write_bytes(b"other"),
     "one: first-dark.mp4 does not match its manifest"),
    (lambda out: os.remove(out / "second.gif"), "two: second.gif is missing"),
    (lambda out: os.remove(out / "second-dark.mp4"), "two: second-dark.mp4 is missing"),
    (lambda out: os.remove(out / "first.jpg"),
     "one: first.jpg is missing or older than its video"),
    (lambda out: os.remove(out / "first-dark-thumb.mp4"),
     "one: first-dark-thumb.mp4 is missing or older than its video"),
    (lambda out: os.utime(out / "second-thumb.mp4", (5, 5)),
     "two: second-thumb.mp4 is missing or older than its video"),
    (lambda out: (out / "second-dark.json").write_text((out / "second-dark.json").read_text()
                                                       .replace('"dark"', '"light"')),
     "two: second-dark.json records a light render"),
])
def test_what_stops_a_release(made, break_it, says):
    break_it(made.out)
    with pytest.raises(SystemExit) as stopped:
        _run(made, "--light-only", "--allow-partial")        # neither flag excuses these
    assert "release: not ready" in str(stopped.value) and says in str(stopped.value)


def test_a_render_without_a_video_or_a_gif_stops_the_release(made):
    manifest = json.loads((made.out / "first.json").read_text())
    del manifest["outputs"]["first.mp4"]
    (made.out / "first.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="one: no MP4 in its light render"):
        _run(made)


def test_the_file_cap_and_a_used_directory(made, monkeypatch):
    made.dest.mkdir()
    (made.dest / "old").write_text("x")
    with pytest.raises(SystemExit, match="is not empty"):
        _run(made)
    os.remove(made.dest / "old")
    monkeypatch.setattr(release, "FILE_CAP", 8)              # this release is 9 files
    with pytest.raises(SystemExit, match="9 files, and a release holds at most 8"):
        _run(made)
    assert os.listdir(made.dest) == []


def test_the_real_set_fits_a_release():
    """Every animation in both themes: a video each, four archives and the checksums."""
    assert 2 * len(presets.REGISTRY) + 5 <= release.FILE_CAP

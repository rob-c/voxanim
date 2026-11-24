"""The website generator: every animation listed once, the page well formed, its media
taken from the release's site archive and its videos streamed from the release."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
from html.parser import HTMLParser

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "website"))
import build_site as site

from voxanim import presets


def test_every_animation_is_listed_once():
    site.check_sections()


def test_check_sections_names_the_problem():
    with pytest.raises(site.SiteError, match="missing \\['conv2d'\\]"):
        site.check_sections([("a", "A", "", [])], {"conv2d": None})
    with pytest.raises(site.SiteError, match="duplicated \\['x'\\]"):
        site.check_sections([("a", "A", "", [("x", "X"), ("x", "X")])], {"x": None})
    with pytest.raises(site.SiteError, match="unknown \\['y'\\]"):
        site.check_sections([("a", "A", "", [("y", "Y")])], {})


def test_prose():
    assert site.prose("7x7x8 -> 5x5x8, 10 steps x 4 channels") == \
        "7×7×8 → 5×5×8, 10 steps × 4 channels"
    assert site.prose("max-pool, dense -> softmax") == "max-pool, dense → softmax"


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.clips, self.frames, self.posters, self.tags = [], [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if "id" in a:
            self.ids.append(a["id"])
        if "data-clip" in a:
            self.clips.append(json.loads(a["data-clip"]))
        if "data-thumb" in a:
            self.frames.append(a)
        if tag == "picture":
            self.posters.append(a)


BOTH = {"seconds": 30.0, "dark": True, "dark_card": True}
LIGHT = {"seconds": 30.0, "dark": False, "dark_card": False}
RELEASE = "https://github.com/%s-%%s/releases/download/v9.9.9/" % site.REPO   # % theme


def _page(found, **kw):
    page = site.page_html(found, "v9.9.9", **kw)
    p = _Collect()
    p.feed(page)
    return page, p


def test_page():
    page, p = _page(dict.fromkeys(presets.REGISTRY, BOTH))
    assert not re.search(r"\{\{\w+\}\}", page)              # every field filled in
    assert len(p.ids) == len(set(p.ids))                    # ids unique
    assert p.clips[0]["name"] == site.HERO                   # the hero is a clip too
    assert [c["name"] for c in p.clips[1:]] == \
        [n for *_, items in site.SECTIONS for n, _ in items]
    for c, frame, poster in zip(p.clips, p.frames, p.posters, strict=True):
        media = presets.REGISTRY[c["name"]].media
        # the full videos are the release's files, one per theme, and nothing on the site
        assert c["video"] == {"light": RELEASE % "light" + media + ".mp4",
                              "dark": RELEASE % "dark" + media + "-dark.mp4"}
        # a card carries the poster and the small clip of both themes, from the site
        assert c["poster"] == {"light": "media/%s.jpg" % media,
                               "dark": "media/%s-dark.jpg" % media}
        assert c["thumb"] == {"light": "media/%s-thumb.mp4" % media,
                              "dark": "media/%s-dark-thumb.mp4" % media}
        assert frame["data-thumb"] == c["thumb"]["light"]
        assert frame["data-thumb-dark"] == c["thumb"]["dark"]
        assert poster["data-dark"] == c["poster"]["dark"]
        assert c["manifest"] == {"light": "media/%s.json" % media,
                                 "dark": "media/%s-dark.json" % media}
        assert c["seconds"] == 30 and c["about"]
    # the dark poster is chosen by the system's colour scheme before any script runs
    sources = [a for tag, a in p.tags if tag == "source" and "srcset" in a]
    assert [a["srcset"] for a in sources] == [c["poster"]["dark"] for c in p.clips]
    assert all(a["media"] == "(prefers-color-scheme: dark)" for a in sources)
    imgs = [a["src"] for tag, a in p.tags if tag == "img" and a["src"].startswith("media/")]
    assert imgs == [c["poster"]["light"] for c in p.clips]
    # one lazy poster per grid clip (the hero's loads eagerly); app.js adds the
    # videos, so the page works without it; the cue to click on every clip
    grid = len(p.clips) - 1
    assert page.count('loading="lazy"') == grid and "<video" not in page.split("<dialog")[0]
    assert page.count('class="open"') == grid + 1 and page.count("data-clip=") == grid + 1
    # no full video and no 480 px copy is on the site: only the small clips
    assert "-480" not in page
    assert not re.search(r'"media/[^"]*(?<!-thumb)\.mp4"', page)


def test_the_player_streams_from_the_release():
    page, _p = _page(dict.fromkeys(presets.REGISTRY, BOTH))
    player = page.split("<dialog")[1]
    q = _Collect()
    q.feed("<dialog" + player)
    # Safari plays a release file only when the page states its type; no bare src
    video, = [a for tag, a in q.tags if tag == "video"]
    assert "src" not in video
    assert [a for tag, a in q.tags if tag == "source"] == [{"type": "video/mp4"}]
    # a way to pick the other theme of this video, and both downloads
    assert [a["data-video-theme"] for _t, a in q.tags if "data-video-theme" in a] == \
        ["light", "dark"]
    assert [a["data-video"] for _t, a in q.tags if "data-video" in a] == ["light", "dark"]
    links = [a.get("href") for tag, a in q.tags if tag == "a"]
    assert RELEASE % "light" + "voxanim-gifs-light.tar" in links
    assert RELEASE % "dark" + "voxanim-gifs-dark.tar" in links
    assert 'class="player-error"' in player and " hidden>" in player
    # the script retries a failed stream once, then shows the message
    with open(os.path.join(os.path.dirname(site.__file__), "app.js"), encoding="utf-8") as fh:
        js = fh.read()
    assert 'source.addEventListener("error", streamFailed)' in js
    assert 'big.addEventListener("error", streamFailed)' in js
    assert "fetch(" not in js                                # the release sends no CORS header


def test_a_card_without_dark_media_keeps_the_light():
    """Dark is offered only where the release has it: the video by its dark manifest, the
    card by its dark poster and clip."""
    names = list(presets.REGISTRY)
    found = dict.fromkeys(names, BOTH)
    found[names[3]] = LIGHT                                   # light only
    found[names[4]] = {"seconds": 30.0, "dark": True, "dark_card": False}
    page, p = _page(found)
    by = {c["name"]: c for c in p.clips}
    plain, video_only, both = by[names[3]], by[names[4]], by[names[5]]
    assert set(plain["video"]) == set(plain["poster"]) == set(plain["thumb"]) == {"light"}
    assert set(plain["manifest"]) == {"light"}
    assert set(video_only["video"]) == {"light", "dark"} and set(video_only["poster"]) == {"light"}
    assert set(both["poster"]) == set(both["thumb"]) == {"light", "dark"}
    for name in (names[3], names[4]):
        card = page.split('<li class="clip" id="%s">' % name)[1].split("</li>")[0]
        assert "data-thumb-dark" not in card and "<picture" not in card
        assert '<img src="media/%s.jpg"' % presets.REGISTRY[name].media in card
    assert "data-thumb-dark" in page.split('<li class="clip" id="%s">' % names[5])[1].split(
        "</li>")[0]
    # with no dark render anywhere, the dark GIF archive is not offered
    assert '<li hidden><a href="%svoxanim-gifs-dark.tar"' % (RELEASE % "dark") in \
        site.page_html(dict.fromkeys(names, LIGHT), "v9.9.9")
    assert "<li hidden>" not in page


def test_the_next_catalogue_is_listed_under_its_own_titles():
    from voxanim.next_catalog import ITEMS, drawing
    listed = {n: t for *_, items in site.SECTIONS for n, t in items}
    anchors = [anchor for anchor, *_ in site.SECTIONS]
    assert len(anchors) == len(set(anchors))
    assert len(site.NEXT_SECTIONS) == 30
    new = len(site.NEXT_SECTIONS)
    for item in ITEMS:
        assert listed[item["preset"]] == item["title"] == drawing(item["name"]).TITLE
        first, about = site.describe(presets.REGISTRY[item["preset"]])
        assert first == site.prose(item["summary"])           # the card shows the plain summary
        assert item["description"].split()[0] in about         # the dialog adds what was measured
    assert new == 30
    for _anchor, heading, intro, items in site.SECTIONS:
        assert heading and intro.endswith(".") and items
        assert "review" not in intro.lower()


def test_the_page_is_ordered_as_one_course():
    """Statistics first, then trees, Gaussian processes, basic networks, autoencoders and
    generative models, recurrent networks, attention, language models, and what follows."""
    anchors = [anchor for anchor, *_ in site.SECTIONS]
    assert anchors == [a for *_, part in site.PARTS for a in part]
    assert len(site.PARTS) == 13 and all(intro.endswith(".") for _t, intro, _a in site.PARTS)
    where = {n: k for k, (*_, items) in enumerate(site.SECTIONS) for n, _ in items}
    route = ["bayes-theorem-areas", "chi-square-fit", "bump-hunt", "decision-tree",
             "gp-prior", "mlp-1-forward-layer1", "conv2d", "cpu-and-gpu-training",
             "autoencoder-dense-2d", "vae-dense-2d", "generative-objectives",
             "rnn-training-step", "lstm-training-step", "embedding-lookup",
             "attention-training-step", "gpt2-attention", "sinusoidal-positions",
             "causal-vs-bidirectional", "gpt2-beam-search", "gpt2-hallucination",
             "mamba-selective-scan"]
    order = [where[n] for n in route]
    assert order == sorted(order)
    assert anchors[0] == "probability" and anchors[-1] == "modern"
    # the animations that changed section to make the order readable
    section = {n: a for a, _h, _i, items in site.SECTIONS for n, _ in items}
    assert section["decision-tree"] == "classical-ml"
    assert section["lstm-training-step"] == section["gated-memory"] == "recurrent"
    assert section["attention-training-step"] == "transformers"
    # nothing before what it rests on
    assert section["cnn-training-step"] == section["strided-cnn-classifier"] == "vision"
    assert section["gpt2-rotary-embeddings"] == section["alibi"] == "positions"
    assert section["weight-entropy"] == "capacity"
    first = {a: items[0][0] for a, _h, _i, items in site.SECTIONS}
    assert first["trees"] == "entropy-and-gini-by-hand"
    assert first["modern"] == "context-cost" and first["latent"] == "pca-and-kmeans"
    assert first["systems"] == "cpu-and-gpu-training"
    # each bridge opens the section it leads into
    assert first["classical-ml"] == "from-a-cut-to-a-tree"
    assert first["convolutions"] == "dense-layer-learns-local"
    assert first["recurrent"] == "markov-chain-to-rnn"
    assert first["rl"] == "labels-against-rewards"
    assert first["efficiency"] == "bits-in-a-weight"
    assert section["every-loss-is-a-likelihood"] == "classical-ml"
    assert section["kl-divergence"] == "information"
    # the counting animations beside the networks they count
    held = {a: [n for n, _ in items] for a, _h, _i, items in site.SECTIONS}
    conv = held["convolutions"]
    assert conv[:2] == ["dense-layer-learns-local", "counting-weights-dnn-and-cnn"]
    tr = held["transformers"]
    assert tr[tr.index("gpt2-transformer-block") + 1] == "counting-weights-transformer"
    # each part is announced once on the page, ahead of its first section
    names = list(presets.REGISTRY)
    page = site.page_html(dict.fromkeys(names, BOTH), "v9.9.9")
    assert page.count('class="part"') == len(site.PARTS)
    # the paragraph under the headline says how many parts, and nothing is left unfilled
    assert "one course in %d parts" % len(site.PARTS) in page and "{{" not in page
    for k, (title, _intro, part) in enumerate(site.PARTS):
        at = page.index('id="part-%d"' % (k + 1))
        assert at < page.index('<section class="group" id="%s"' % part[0])
        assert site.html.escape(title) in page[at:at + 400]


def test_an_animation_without_media_in_the_release_is_left_off_the_page():
    """A push between releases must not break the site or show a clip with nothing to play: the
    Pages build takes its media from the latest release, which the newest animations are not in
    yet."""
    names = list(presets.REGISTRY)
    held_back = set(names[128:])
    found = {n: BOTH for n in names if n not in held_back}
    page, p = _page(found)
    shown = {c["name"] for c in p.clips}
    assert shown == set(found) and not shown & held_back
    assert "%d animations" % len(found) in page
    for anchor, _heading, _intro, items in site.SECTIONS:      # a section with nothing to show
        assert (anchor in p.ids) == any(n in found for n, _ in items)
    assert page.count('class="part"') < len(site.PARTS)        # nor a part with no section
    # every section survives when everything has media
    assert len(site.released(site.SECTIONS, dict.fromkeys(names, BOTH))) == len(site.SECTIONS)
    with pytest.raises(site.SiteError, match="no animation has"):
        site.page_html({}, "v9.9.9")


# ------------------------------------------------------------------ the media
# Stand-ins for a render folder and a release: the builder only moves files and reads the
# manifests, so their contents do not matter.

SHOWN = ("conv2d", "bump-hunt")                                # bump-hunt: light only


def _renders(folder, videos=True):
    folder.mkdir()
    for name in SHOWN:
        media = presets.REGISTRY[name].media
        for theme in ("", "-dark") if name == "conv2d" else ("",):
            (folder / (media + theme + ".jpg")).write_bytes(b"poster" + theme.encode())
            (folder / (media + theme + "-thumb.mp4")).write_bytes(b"clip" + theme.encode())
            (folder / (media + theme + ".json")).write_text(json.dumps({"fps": 30, "frames": 45}))
            if videos:
                (folder / (media + theme + ".mp4")).write_bytes(b"video" + theme.encode())
    return folder


def _archive(path, folder, extra=()):
    with tarfile.open(path, "w:gz") as tf:
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".mp4") or f.endswith("-thumb.mp4"):
                tf.add(folder / f, arcname=f)
        for name in extra:
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    return path


def test_the_build_takes_everything_from_the_site_archive(tmp_path, monkeypatch, capsys):
    """From a release: one download (the archive), no ffmpeg, no video on the site."""
    archive = _archive(tmp_path / site.SITE_ARCHIVE, _renders(tmp_path / "renders"))
    asked = []

    def gh(cmd, **kw):
        asked.append(cmd)
        assert cmd[:3] == ["gh", "release", "download"] and cmd[3] == "v9.9.9"
        assert cmd[cmd.index("--pattern") + 1] == site.SITE_ARCHIVE
        os.link(archive, os.path.join(cmd[cmd.index("--dir") + 1], site.SITE_ARCHIVE))
        return site.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(site.subprocess, "run", gh)          # the only command it may run
    out = tmp_path / "site"
    found = site.build(str(out), tag="v9.9.9")
    assert len(asked) == 1
    assert found == {"conv2d": {"seconds": 1.5, "dark": True, "dark_card": True},
                     "bump-hunt": {"seconds": 1.5, "dark": False, "dark_card": False}}
    conv = presets.REGISTRY["conv2d"].media
    assert sorted(os.listdir(out / "media")) == sorted(
        [conv + t for t in (".jpg", "-thumb.mp4", ".json", "-dark.jpg", "-dark-thumb.mp4",
                            "-dark.json")]
        + ["bump-hunt" + t for t in (".jpg", "-thumb.mp4", ".json")])
    for f in (*site.STATIC, "index.html", ".nojekyll"):
        assert (out / f).exists()
    p = _Collect()
    p.feed((out / "index.html").read_text(encoding="utf-8"))
    clips = {c["name"]: c for c in p.clips}
    assert set(clips) == set(SHOWN)
    assert clips["conv2d"]["video"] == {"light": RELEASE % "light" + conv + ".mp4",
                                        "dark": RELEASE % "dark" + conv + "-dark.mp4"}
    assert clips["bump-hunt"]["video"] == {"light": RELEASE % "light" + "bump-hunt.mp4"}
    assert clips["bump-hunt"]["seconds"] == 2                # 45 frames at 30 a second
    for c in clips.values():                                 # what a card names is there
        for f in (*c["poster"].values(), *c["thumb"].values(), *c["manifest"].values()):
            assert (out / f).exists()
    assert "%d animations have no media in release v9.9.9" % (len(presets.REGISTRY) - 2) \
        in capsys.readouterr().out


def test_a_release_without_the_archive_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(site.subprocess, "run", lambda cmd, **kw: site.subprocess.
                        CompletedProcess(cmd, 1, "", "no assets match the file pattern"))
    with pytest.raises(site.SiteError, match=r"release v1\.10\.0 has no voxanim-site\.tar\.gz"):
        site.build(str(tmp_path / "site"), tag="v1.10.0")


def test_only_plain_media_files_are_unpacked(tmp_path):
    renders = _renders(tmp_path / "renders")
    for bad in ("../outside.jpg", "sub/dir.jpg", ".hidden.json", "index.html", "app.js"):
        archive = _archive(tmp_path / "bad.tar.gz", renders, extra=[bad])
        with pytest.raises(site.SiteError, match="should not"):
            site.unpack_site(str(archive), str(tmp_path / "dest"))
    assert not (tmp_path / "outside.jpg").exists()
    good = _archive(tmp_path / "good.tar.gz", renders)
    assert site.unpack_site(str(good), str(tmp_path / "ok")) == 9


@pytest.mark.parametrize("packed", [False, True])
def test_a_local_preview_plays_local_files(tmp_path, monkeypatch, packed):
    """``--media DIR`` (a render folder, or one gathered for a release) needs no release, no
    GitHub CLI and no ffmpeg, and the player gets DIR's own videos."""
    def nothing(cmd, **kw):
        raise AssertionError("ran %s" % cmd)
    monkeypatch.setattr(site.subprocess, "run", nothing)
    folder = _renders(tmp_path / "renders")
    if packed:                                               # as tools/release_assets.py leaves it
        _archive(tmp_path / site.SITE_ARCHIVE, folder)
        for d in ("voxanim", "voxanim-light", "voxanim-dark"):
            (folder / d).mkdir()
        os.rename(tmp_path / site.SITE_ARCHIVE, folder / "voxanim" / site.SITE_ARCHIVE)
        for f in os.listdir(folder):
            if f.endswith((".jpg", ".json", "-thumb.mp4")):
                os.remove(folder / f)
            elif f.endswith(".mp4"):
                os.rename(folder / f, folder / ("voxanim-dark" if "-dark" in f
                                                else "voxanim-light") / f)
    out = tmp_path / "site"
    found = site.build(str(out), media_dir=str(folder))
    assert set(found) == set(SHOWN) and found["conv2d"]["dark"] and not found["bump-hunt"]["dark"]
    p = _Collect()
    p.feed((out / "index.html").read_text(encoding="utf-8"))
    conv = presets.REGISTRY["conv2d"].media
    clips = {c["name"]: c for c in p.clips}
    assert clips["conv2d"]["video"] == {"light": "media/%s.mp4" % conv,
                                        "dark": "media/%s-dark.mp4" % conv}
    assert (out / "media" / (conv + "-dark.mp4")).read_bytes() == b"video-dark"
    assert (out / "media" / "bump-hunt.mp4").read_bytes() == b"video"
    assert p.clips[0]["name"] == "conv2d"                    # the hero


def test_a_local_preview_without_the_hero_leads_with_what_it_has(tmp_path):
    folder = _renders(tmp_path / "renders")
    for f in os.listdir(folder):
        if not f.startswith("bump-hunt"):
            os.remove(folder / f)
    out = tmp_path / "site"
    assert set(site.build(str(out), media_dir=str(folder))) == {"bump-hunt"}
    p = _Collect()
    p.feed((out / "index.html").read_text(encoding="utf-8"))
    assert [c["name"] for c in p.clips] == ["bump-hunt", "bump-hunt"]
    # and an animation whose video is not in the folder cannot be played, so it is left off
    os.remove(folder / "bump-hunt.mp4")
    with pytest.raises(site.SiteError, match="no animation has"):
        site.build(str(tmp_path / "site2"), media_dir=str(folder))

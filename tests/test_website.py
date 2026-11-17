"""The website generator: every animation listed once, the page well formed."""

from __future__ import annotations

import json
import os
import re
import sys
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
        self.ids, self.clips = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if "data-clip" in a:
            self.clips.append(json.loads(a["data-clip"]))


def test_page():
    durations = dict.fromkeys(presets.REGISTRY, 30.0)
    page = site.page_html(durations, "v9.9.9")
    assert not re.search(r"\{\{\w+\}\}", page)              # every field filled in
    p = _Collect()
    p.feed(page)
    assert len(p.ids) == len(set(p.ids))                    # ids unique
    assert p.clips[0]["name"] == site.HERO                   # the hero is a clip too
    assert [c["name"] for c in p.clips[1:]] == \
        [n for *_, items in site.SECTIONS for n, _ in items]
    for c in p.clips:
        media = presets.REGISTRY[c["name"]].media
        assert c["mp4"] == "media/%s.mp4" % media
        assert c["preview"] == "media/%s-480.mp4" % media and c["poster"] == "media/%s.jpg" % media
        assert c["thumb"] == "media/%s-thumb.mp4" % media
        assert c["gif"].endswith("/releases/download/v9.9.9/%s.gif" % media)
        assert c["seconds"] == 30 and c["about"]
    # one lazy poster per grid clip (the hero's loads eagerly); app.js adds the
    # videos, so the page works without it; the cue to click on every clip
    grid = len(p.clips) - 1
    assert page.count('loading="lazy"') == grid and "<video" not in page.split("<dialog")[0]
    assert page.count('class="open"') == grid + 1 and page.count("data-clip=") == grid + 1


def test_the_next_catalogue_is_listed_under_its_own_titles():
    from voxanim.next_catalog import ITEMS, drawing
    listed = {n: t for *_, items in site.SECTIONS for n, t in items}
    anchors = [anchor for anchor, *_ in site.SECTIONS]
    assert len(anchors) == len(set(anchors))
    assert len(site.NEXT_SECTIONS) == 27
    new = len(site.NEXT_SECTIONS)
    for item in ITEMS:
        assert listed[item["preset"]] == item["title"] == drawing(item["name"]).TITLE
        first, about = site.describe(presets.REGISTRY[item["preset"]])
        assert first == site.prose(item["summary"])           # the card shows the plain summary
        assert item["description"].split()[0] in about         # the dialog adds what was measured
    assert new == 27
    for _anchor, heading, intro, items in site.SECTIONS:
        assert heading and intro.endswith(".") and items
        assert "review" not in intro.lower()


def test_the_page_is_ordered_as_one_course():
    """Statistics first, then trees, basic networks, autoencoders and generative models,
    Gaussian processes, recurrent networks, attention, language models, and what follows."""
    anchors = [anchor for anchor, *_ in site.SECTIONS]
    assert anchors == [a for *_, part in site.PARTS for a in part]
    assert len(site.PARTS) == 12 and all(intro.endswith(".") for _t, intro, _a in site.PARTS)
    where = {n: k for k, (*_, items) in enumerate(site.SECTIONS) for n, _ in items}
    route = ["bayes-theorem-areas", "chi-square-fit", "bump-hunt", "decision-tree",
             "mlp-1-forward-layer1", "conv2d", "cpu-and-gpu-training", "autoencoder-dense-2d",
             "vae-dense-2d", "generative-objectives", "gp-prior", "rnn-training-step",
             "lstm-training-step", "embedding-lookup", "attention-training-step",
             "gpt2-attention", "causal-vs-bidirectional", "gpt2-beam-search",
             "gpt2-hallucination", "mamba-selective-scan"]
    order = [where[n] for n in route]
    assert order == sorted(order)
    assert anchors[0] == "probability" and anchors[-1] == "modern"
    # the animations that changed section to make the order readable
    section = {n: a for a, _h, _i, items in site.SECTIONS for n, _ in items}
    assert section["decision-tree"] == "classical-ml"
    assert section["lstm-training-step"] == section["gated-memory"] == "recurrent"
    assert section["attention-training-step"] == "transformers"
    # each part is announced once on the page, ahead of its first section
    names = list(presets.REGISTRY)
    page = site.page_html(dict.fromkeys(names, 30.0), "v9.9.9")
    assert page.count('class="part"') == len(site.PARTS)
    # the paragraph under the headline says how many parts, and nothing is left unfilled
    assert "one course in %d parts" % len(site.PARTS) in page and "{{" not in page
    for k, (title, _intro, part) in enumerate(site.PARTS):
        at = page.index('id="part-%d"' % (k + 1))
        assert at < page.index('<section class="group" id="%s"' % part[0])
        assert site.html.escape(title) in page[at:at + 400]


def test_an_animation_without_a_video_in_the_release_is_left_off_the_page():
    """A push between releases must not break the site or show a clip with no video: the Pages
    build takes its videos from the latest release, which the newest animations are not in yet."""
    names = list(presets.REGISTRY)
    held_back = set(names[128:])
    durations = {n: 30.0 for n in names if n not in held_back}
    page = site.page_html(durations, "v9.9.9")
    p = _Collect()
    p.feed(page)
    shown = {c["name"] for c in p.clips}
    assert shown == set(durations) and not shown & held_back
    assert "%d animations" % len(durations) in page
    for anchor, _heading, _intro, items in site.SECTIONS:      # a section with nothing to show
        assert (anchor in p.ids) == any(n in durations for n, _ in items)
    assert page.count('class="part"') < len(site.PARTS)        # nor a part with no section
    # every section survives when everything has a video
    assert len(site.released(site.SECTIONS, dict.fromkeys(names, 1.0))) == len(site.SECTIONS)


def test_local_media_must_be_complete(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "have.mp4").write_bytes(b"x")
    with pytest.raises(site.SiteError, match="no video for: missing"):
        site.fetch_media(str(tmp_path / "out"), str(tmp_path / "src"), "v0", ["have", "missing"])
    assert site.fetch_media(str(tmp_path / "out"), str(tmp_path / "src"), "v0", ["have"]) == []

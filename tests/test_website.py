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
    # one lazy poster per clip; app.js adds the videos (the page works without it)
    assert page.count('loading="lazy"') == len(p.clips) and "<video" not in page.split("<dialog")[0]
    # the cue to click on every clip and on the hero (which is a clip too)
    assert page.count('class="open"') == len(p.clips) + 1
    assert page.count("data-clip=") == len(p.clips) + 1

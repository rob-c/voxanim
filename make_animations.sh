#!/usr/bin/env bash
# Render animations and encode them (GIF + MP4): a thin wrapper around
# `python3 -m voxanim render`, run from this directory.
#
#   ./make_animations.sh conv1d-full-width-scroll conv1d-row-sweep --install
#                                        the conv1d pair -> ../media (what the site links)
#   ./make_animations.sh all             every animation -> build/out/
#   ./make_animations.sh all --install   every animation -> ../media (replaces originals)
#   ./make_animations.sh conv2d depthwise --scale 2     any names, any render options
#
# See `python3 -m voxanim --help` and README.md.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -eq 0 ]]; then
    sed -n '2,11s/^# \{0,1\}//p' "$0" >&2
    exit 2
fi
exec python3 -m voxanim render "$@"

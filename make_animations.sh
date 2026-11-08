#!/usr/bin/env bash
# Render animations and encode them (GIF + MP4).
#
#   ./make_animations.sh                 conv1d pair -> ../media (what the site links)
#   ./make_animations.sh all             every animation -> build/out/
#   ./make_animations.sh all --install   every animation -> ../media (replaces originals)
#   ./make_animations.sh conv2d depthwise --scale 2     any names, any render options
#
# See `python3 -m voxanim --help` and README.md.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -eq 0 ]]; then
    exec python3 -m voxanim render conv1d-full-width-scroll conv1d-row-sweep --install
fi
exec python3 -m voxanim render "$@"

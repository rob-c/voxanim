"""Write the logo's vector files for the website from :mod:`voxanim.brand`.

    python3 tools/make_logo.py            # into website/

``logo.svg`` is the V with the name (dark text, for a light page), ``logo-dark.svg`` the same
for a dark page, and ``logo-mark.svg`` the V alone (the page's icon and header).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxanim import brand

FILES = {
    "logo.svg": lambda: brand.svg("lockup", url=False),
    "logo-dark.svg": lambda: brand.svg("lockup", ink="#e8e8e6", url=False),
    "logo-mark.svg": lambda: brand.svg("mark"),
}


def main(out=None):
    out = out or os.path.join(os.path.dirname(__file__), "..", "website")
    for name, make in FILES.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(make() + "\n")
        print("wrote", os.path.join(out, name))


if __name__ == "__main__":
    main(*sys.argv[1:2])

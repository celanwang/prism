"""Prism command-line entry point.

Placeholder until P0 (measurement foundation) lands; see docs/ROADMAP.md.
"""

import argparse

from prism import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    print(f"prism {__version__}: scaffolding only; P0 not started, see docs/ROADMAP.md")


if __name__ == "__main__":
    main()

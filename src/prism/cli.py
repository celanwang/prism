"""Prism command-line entry point.

Placeholder until P0 (measurement foundation) lands; see docs/ROADMAP.md.
"""

import argparse
import json
from pathlib import Path

from prism import __version__
from prism.env import collect_env, env_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    env_parser = subparsers.add_parser("env", help="Describe the experiment environment")
    env_parser.add_argument("--write", action="store_true", help="Write configs/env.lock")
    args = parser.parse_args()
    if args.command == "env":
        env = collect_env()
        output = json.dumps(env, sort_keys=True, indent=2)
        if args.write:
            path = Path(__file__).resolve().parents[2] / "configs" / "env.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output + "\n", encoding="utf-8")
            print(f"{path}\n{env_fingerprint(env)}")
        else:
            print(output)
        return
    print(f"prism {__version__}: scaffolding only; P0 not started, see docs/ROADMAP.md")


if __name__ == "__main__":
    main()

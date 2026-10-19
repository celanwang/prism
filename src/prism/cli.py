"""Prism command-line entry point.

Placeholder until P0 (measurement foundation) lands; see docs/ROADMAP.md.
"""

import argparse
import json
from pathlib import Path

from prism import __version__
from prism.env import collect_env, env_fingerprint
from prism.sut import DEFAULT_SUT_PATH, check_sut


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    env_parser = subparsers.add_parser("env", help="Describe the experiment environment")
    env_parser.add_argument("--write", action="store_true", help="Write configs/env.lock")
    sut_parser = subparsers.add_parser("sut", help="Check the reference model")
    sut_commands = sut_parser.add_subparsers(dest="sut_command", required=True)
    check_parser = sut_commands.add_parser("check", help="Verify artifacts and greedy consistency")
    check_parser.add_argument("--path", type=Path, default=DEFAULT_SUT_PATH)
    args = parser.parse_args()
    if args.command == "sut":
        try:
            result = check_sut(args.path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"SUT check failed: {error}\n")
        print(json.dumps(result, sort_keys=True, indent=2))
        return
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

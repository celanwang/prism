"""Prism command-line entry point.

Repository experiments, provenance verification and isolated reproduction.
"""

import argparse
import json
import sys
from pathlib import Path

from prism import __version__
from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement
from prism.record import save_run, summarize
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
    record_parser = subparsers.add_parser("record", help="Save and summarize experiment records")
    record_commands = record_parser.add_subparsers(dest="record_command", required=True)
    save_parser = record_commands.add_parser("save", help="Save a complete experiment run")
    save_parser.add_argument("--slug", required=True)
    save_parser.add_argument("--config", required=True, type=Path)
    save_parser.add_argument(
        "--measurement", type=Path, help="Measurement JSON file (default: stdin)"
    )
    save_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    summary_parser = record_commands.add_parser(
        "summary", help="Summarize validated experiment runs"
    )
    summary_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    subparsers.add_parser("verify", help="Verify committed artifact SHA-256 inventory")
    repro = subparsers.add_parser("reproduce", help="Reproduce recipes in a repository clone")
    selection = repro.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="List recipes and dependencies")
    selection.add_argument("--all", action="store_true", help="Run all recipes in dependency order")
    selection.add_argument("--sample", action="store_true", help="Run full representative recipes")
    repro.add_argument("--dry-run", action="store_true", help="Plan only; no files or processes")
    repro.add_argument("--timeout", type=float, default=1800, help="Per-process seconds")
    repro.add_argument("-o", "--output", type=Path, help="New JSON under runs/.reproduce")
    repro.add_argument("action", nargs="?", choices=["run"])
    repro.add_argument("recipe", nargs="?")
    args = parser.parse_args()
    if args.command in ("verify", "reproduce"):
        from prism.reproduce import load_recipes, reproduce, verify_inventory

        try:
            if args.command == "verify":
                result = verify_inventory()
            else:
                config = load_recipes()
                if args.action and (args.list or args.all or args.sample):
                    parser.error("run cannot be combined with --list/--all/--sample")
                if args.recipe and not args.action:
                    parser.error("recipe requires run")
                if args.list:
                    if args.output or args.dry_run:
                        parser.error("--list cannot be combined with -o/--dry-run")
                    result = config
                else:
                    if args.action == "run" and not args.recipe:
                        parser.error("run requires a recipe")
                    selected = (
                        list(config["recipes"])
                        if args.all
                        else config["sample"]
                        if args.sample
                        else [args.recipe]
                        if args.recipe
                        else []
                    )
                    if not selected:
                        parser.error("choose --list, --all, --sample or run <recipe>")
                    result = reproduce(
                        selected, dry_run=args.dry_run, output=args.output, timeout=args.timeout
                    )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            parser.exit(1, f"Reproduction failed: {error}\n")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        if not result.get("ok", True):
            parser.exit(1)
        return
    if args.command == "record":
        try:
            if args.record_command == "summary":
                output = summarize(args.runs_dir)
            else:
                config = json.loads(args.config.read_text(encoding="utf-8"))
                value = (
                    args.measurement.read_text(encoding="utf-8")
                    if args.measurement is not None
                    else sys.stdin.read()
                )
                output = save_run(args.runs_dir, args.slug, config, Measurement.from_json(value))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"Record failed: {error}\n")
        print(output)
        return
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
    print(f"prism {__version__}: P0–P4 experiments; see prism reproduce --list")
    parser.print_help()


if __name__ == "__main__":
    main()

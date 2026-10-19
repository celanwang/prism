"""Audit a release candidate in a clean temporary clone and a new uv environment.

--check runs a small process/CLI smoke, never release evidence. The default runs
all tests, every README shell command, and P5-T1's full representative sample.
Only P5-T2's six allowed source files may be snapshotted from an uncommitted
worktree. A second clone of that committed snapshot receives no developer venv,
PYTHONPATH or uv project overrides. Logs, patch and replay records survive cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from prism.reproduce import load_recipes, read_json, safe_path, topological_order, write_json_new

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_FILES = (
    "scripts/release_check.py",
    "configs/release.json",
    "tests/test_release.py",
    "results/p5/REPORT.md",
    "README.md",
    "docs/ROADMAP.md",
)


def clean_env() -> dict[str, str]:
    """Retain OS/tool discovery, discard inherited Python/uv/git/BLAS overrides."""
    prefixes = ("PYTHON", "UV_", "GIT_", "CONDA", "VIRTUAL_ENV")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    for name in (
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, env=clean_env()).decode().strip()


def load_manifest(root: Path, path: Path) -> dict:
    config = read_json(path)
    if config.get("version") != 1 or config.get("candidate_version") != "0.1.0":
        raise ValueError("invalid release version")
    timeout = config.get("timeout_seconds")
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    phases = config.get("phases")
    if not isinstance(phases, dict) or set(phases) != {f"P{i}" for i in range(6)}:
        raise ValueError("phases must cover P0–P5")
    for phase, entry in phases.items():
        if not isinstance(entry, dict) or not entry.get("reports") or not entry.get("prs"):
            raise ValueError(f"missing evidence: {phase}")
        for name in entry["reports"]:
            if not isinstance(name, str) or not name.startswith(f"results/{phase.lower()}/"):
                raise ValueError(f"invalid report: {name}")
            safe_path(root, name)
        if any(type(n) is not int or n <= 0 for n in entry["prs"]):
            raise ValueError(f"invalid PR: {phase}")
    return config


def readme_commands(text: str) -> list[list[str]]:
    commands = []
    for block in re.findall(r"```(?:sh|bash)\n(.*?)```", text, re.S):
        for line in block.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            argv = shlex.split(line)
            if argv[:3] != ["uv", "run", "--locked"] and argv != ["uv", "sync", "--locked"]:
                raise ValueError(f"unsupported README command: {line}")
            if any(c in line for c in ("|", ";", "&&", "$", "`", ">", "<", "\\")):
                raise ValueError(f"README commands must be standalone argv: {line}")
            commands.append(argv)
    if not commands:
        raise ValueError("no README shell commands")
    return commands


def audit_documents(root: Path, config: dict) -> dict:
    """Check evidence pointers; semantic limitations are reviewed in the P5 report."""
    roadmap = (root / "docs/ROADMAP.md").read_text()
    inventory = read_json(root / "artifacts/inventory-v1.json")
    covered = {entry["path"] for entry in inventory["files"]}
    history = git(root, "log", "--format=%s")
    rows, errors = [], []
    for phase, evidence in config["phases"].items():
        lines = [line for line in roadmap.splitlines() if line.startswith(f"| {phase} ")]
        if len(lines) != 1 or "完成" not in lines[0]:
            errors.append(f"{phase}: missing completed/candidate-reviewed ROADMAP row")
        for name in evidence["reports"]:
            path = safe_path(root, name)
            ok = path.is_file() and path.stat().st_size > 0
            pointer = (
                name in lines[0] or f"results/{phase.lower()}/" in lines[0] if lines else False
            )
            immutable = name in covered
            rows.append(
                {
                    "phase": phase,
                    "report": name,
                    "exists": ok,
                    "roadmap_pointer": pointer,
                    "inventory_covered": immutable,
                }
            )
            if not ok or not pointer or (phase != "P5" and not immutable):
                errors.append(f"{phase}: evidence/pointer/inventory mismatch: {name}")
        for number in evidence["prs"]:
            if not re.search(rf"Merge pull request #{number}\b", history):
                errors.append(f"{phase}: no local merge evidence for PR #{number}")
    agents = (root / "AGENTS.md").read_text()
    design = (root / "DESIGN.md").read_text()
    for value in ("Codex", "Kimi", "uv run --locked pytest", "git push"):
        if value not in agents:
            errors.append(f"AGENTS convention missing: {value}")
    if "无 CUDA" not in design or "模拟" not in design:
        errors.append("DESIGN hardware boundary missing")
    return {
        "ok": not errors,
        "errors": errors,
        "phases": rows,
        "readme_commands": readme_commands((root / "README.md").read_text()),
        "scope": (
            "Pointers and merge history checked mechanically; "
            "semantic review in results/p5/REPORT.md."
        ),
    }


def review_comparison(report: dict, config: dict, *, check: bool) -> dict:
    """Fail closed for missing recipes/anchors/hashes; env drift remains explicit."""
    expected = topological_order(config["recipes"], ["sut"] if check else config["sample"])
    recipes = report.get("recipes", [])
    errors, rows = [], []
    if not report.get("ok") or [r.get("name") for r in recipes] != expected:
        errors.append("failed or incomplete recipe order")
    for recipe in recipes:
        name = recipe["name"]
        if not recipe.get("ok") or recipe.get("missing_anchors"):
            errors.append(f"{name}: failed/missing anchors")
        records = recipe.get("records", [])
        matched = [r for r in records if "anchor" in r]
        anchors = config["recipes"].get(name, {}).get("anchors", [])
        if sorted(r["anchor"] for r in matched) != sorted(anchors):
            errors.append(f"{name}: incomplete anchor coverage")
        for row in matched:
            if row.get("config_equal") is not True or row.get("config_hash") != row.get(
                "historical_config_hash"
            ):
                errors.append(f"{name}: config mismatch")
            if type(row.get("env_equal")) is not bool or row["env_equal"] != (
                row.get("env_fingerprint") == row.get("historical_env_fingerprint")
            ):
                errors.append(f"{name}: absent/inconsistent env comparison")
            if row.get("deterministic_equal") is False:
                errors.append(f"{name}: deterministic output mismatch")
            if "drift_ms" in row and set(row.get("t4_annotation", {})) != {"p50", "p95", "p99"}:
                errors.append(f"{name}: missing T4 annotations")
        if name == "calibrate" and recipe.get("report_config_equal") is not True:
            errors.append("calibrate: report config mismatch")
        rows.append(
            {
                "name": name,
                "anchors": len(matched),
                "env_differences": sum(r.get("env_equal") is False for r in matched),
                "unanchored": sum(r.get("status") == "unanchored" for r in records),
                "new_derived": sum(r.get("status") == "new-derived" for r in records),
                "t4_above": {
                    q: sum(r.get("t4_annotation", {}).get(q) == "above-resolution" for r in matched)
                    for q in ("p50", "p95", "p99")
                },
            }
        )
    return {
        "ok": not errors,
        "errors": errors,
        "recipes": rows,
        "smoke_only": check,
        "t4": report.get("t4"),
    }


class Runner:
    def __init__(self, output: Path, timeout: float):
        self.output, self.timeout, self.steps = output, timeout, []

    def run(self, argv: list[str], cwd: Path) -> None:
        log = self.output / f"{len(self.steps):02d}.log"
        step = {"argv": argv, "cwd": str(cwd), "log": log.name}
        self.steps.append(step)
        write_json_new(self.output / f"{len(self.steps) - 1:02d}.json", step)
        print(f"[{len(self.steps)}] {shlex.join(argv)}", flush=True)
        start = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=clean_env(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                step["returncode"] = process.wait(timeout=self.timeout)
                if step["returncode"]:
                    raise RuntimeError(f"exit {step['returncode']}: {shlex.join(argv)}; {log}")
            finally:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                process.wait()
                step["elapsed_seconds"] = time.monotonic() - start


def prepare_clone(root: Path, temporary: Path, output: Path, runner: Runner) -> tuple[Path, dict]:
    changed = git(root, "diff", "--name-only", "HEAD").splitlines()
    changed += git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    for name in changed:
        if name not in SNAPSHOT_FILES and not re.fullmatch(r"runs/\d{8}-p5-release-[^/]+/.*", name):
            raise ValueError(f"dirty file outside release whitelist: {name}")
        if not (root / name).is_file():
            raise ValueError(f"deleted/missing source: {name}")
    base = git(root, "rev-parse", "HEAD")
    temporary = temporary.resolve()
    seed, clone = temporary / "seed", temporary / "repo"
    runner.run(["git", "clone", "--no-local", str(root), str(seed)], temporary)
    if git(seed, "rev-parse", "HEAD") != base:
        raise ValueError("source HEAD changed while cloning")
    fingerprints = {}
    for name in SNAPSHOT_FILES:
        source = safe_path(root, name)
        if source.is_file():
            raw = source.read_bytes()
            target = seed / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            archived = output / "candidate" / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(raw)
            fingerprints[name] = hashlib.sha256(raw).hexdigest()
    runner.run(["git", "add", "--", *fingerprints], seed)
    if git(seed, "diff", "--cached", "--name-only"):
        runner.run(
            [
                "git",
                "-c",
                "user.name=Prism release snapshot",
                "-c",
                "user.email=release@prism.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-m",
                "Snapshot P5-T2 candidate inputs for independent replay",
            ],
            seed,
        )
    snapshot = git(seed, "rev-parse", "HEAD")
    patch = subprocess.check_output(
        ["git", "diff", "--binary", base, snapshot], cwd=seed, env=clean_env()
    )
    (output / "candidate.patch").write_bytes(patch)
    runner.run(["git", "clone", "--no-local", str(seed), str(clone)], temporary)
    if git(clone, "status", "--porcelain") or (clone / ".venv").exists():
        raise ValueError("clone is not clean or inherited a venv")
    return clone, {
        "base_commit": base,
        "snapshot_commit": snapshot,
        "snapshot_tree": git(clone, "rev-parse", "HEAD^{tree}"),
        "candidate_sha256": fingerprints,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "clone_clean_before_sync": True,
        "venv_absent_before_sync": True,
    }


def archive_replays(clone: Path, output: Path) -> None:
    base = clone / "runs/.reproduce"
    if base.exists():
        with tarfile.open(output / "reproduction.tar.gz", "w:gz") as archive:
            for path in sorted(base.rglob("*")):
                parts = path.relative_to(base).parts
                # Keep raw measurements, generated reports, comparison JSON and logs;
                # immutable code/config/SUT copies are recoverable from git + patch.
                if path.is_file() and (len(parts) <= 2 or parts[2] in ("runs", "results")):
                    archive.add(path, arcname=str(Path("reproduction") / path.relative_to(base)))
    for child in clone.glob("runs/*-p5-release-*"):
        shutil.copytree(child, output / "readme-check" / child.name)


def release_check(root: Path = ROOT, *, check: bool = False, runs_dir: Path | None = None) -> Path:
    config = load_manifest(root, root / "configs/release.json")
    stamp = datetime.now(UTC).strftime("%Y%m%d-p5-release-%H%M%S%f")
    output = (runs_dir or root / "runs") / (stamp + ("-check" if check else "-full"))
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "config.json", config)
    runner = Runner(output, config["timeout_seconds"])
    result = {
        "ok": False,
        "smoke_only": check,
        "steps": runner.steps,
        "started_at": datetime.now(UTC).isoformat(),
    }
    clone = None
    # Keep the directory alive until every failure log and partial replay is copied.
    with tempfile.TemporaryDirectory(prefix="prism-release-") as directory:
        try:
            clone, result["source"] = prepare_clone(root, Path(directory), output, runner)
            commands = [
                ["uv", "sync", "--locked"],
                ["uv", "run", "--locked", "pytest", "tests/test_cli.py"]
                if check
                else ["uv", "run", "--locked", "pytest"],
                ["uv", "run", "--locked", "prism", "verify"],
            ]
            if not check:
                result["documents"] = audit_documents(clone, config)
                if not result["documents"]["ok"]:
                    raise ValueError(f"document mismatch: {result['documents']['errors']}")
                commands.extend(result["documents"]["readme_commands"])
            replay = ["uv", "run", "--locked", "prism", "reproduce"]
            replay += ["run", "sut"] if check else ["--sample"]
            commands.append(replay)
            executed = set()
            for command in commands:
                key = tuple(command)
                if key not in executed:
                    runner.run(command, clone)
                    executed.add(key)
            reports = sorted((clone / "runs/.reproduce").glob("*/report.json"))
            if len(reports) != 1:
                raise ValueError(f"expected one P5-T1 replay report, got {len(reports)}")
            report = read_json(reports[0])
            write_json_new(output / "reproduction.json", report)
            result["comparison"] = review_comparison(report, load_recipes(clone), check=check)
            if not result["comparison"]["ok"]:
                raise ValueError(f"comparison failed: {result['comparison']['errors']}")
            runner.run(["git", "status", "--short"], clone)
            result["clone_status_after"] = git(clone, "status", "--porcelain")
            if result["clone_status_after"]:
                raise ValueError("clone changed tracked/unignored files")
            result["ok"] = True
        except (Exception, KeyboardInterrupt) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                if clone is not None:
                    archive_replays(clone, output)
            except OSError as error:
                result.update(ok=False, archive_error=str(error))
            result["finished_at"] = datetime.now(UTC).isoformat()
            write_json_new(output / "release.json", result)
    print(
        json.dumps(
            {"ok": result["ok"], "smoke_only": check, "output": str(output)}, ensure_ascii=False
        ),
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Small clone/CLI smoke; not release evidence"
    )
    parser.add_argument(
        "--runs-dir", type=Path, help="Parent for a new timestamped evidence directory"
    )
    args = parser.parse_args()
    try:
        output = release_check(check=args.check, runs_dir=args.runs_dir)
        if not read_json(output / "release.json")["ok"]:
            parser.exit(1, f"Release check failed: {output / 'release.json'}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    main()

"""Reproduction CLI for a repository clone (not a standalone wheel).

Usage: prism verify; prism reproduce --list; prism reproduce run baseline;
prism reproduce --all --dry-run; prism reproduce --sample [-o NEW/report.json].
Outputs are excluded from inventory and git by default. Every recipe runs in a
fresh private tree under runs/.reproduce; original scripts/configs/SUT are copied
without changing the measurement protocol. Dependencies order recipes; historical
service models and the T4 threshold stay frozen, rather than being silently
replaced by this machine's new fit. Historical Markdown is not rebuilt bytewise.
The wheel contains Python modules only: scripts, configs, inventory and SUT must
be supplied by a repository clone. See configs/reproduce.json for anchor limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
THREADS = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"invalid relative path: {name!r}")
    part = PurePosixPath(name)
    if part.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"invalid relative path: {name!r}")
    path = root / name
    # Reject symlinks even when their target is within root.
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError(f"symlink not allowed: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path outside root: {name}")
    return path


def load_inventory(path: Path) -> dict:
    inventory = read_json(path)
    if inventory.get("version") != 1 or not re.fullmatch(
        r"[0-9a-f]{40}", inventory.get("source_commit", "")
    ):
        raise ValueError("invalid inventory version/source_commit")
    entries = inventory.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("empty inventory")
    seen = set()
    for entry in entries:
        name = entry["path"]
        safe_path(path.parent, name)
        if name in seen or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"duplicate path or invalid digest: {name}")
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise ValueError(f"invalid size: {name}")
        seen.add(name)
    return inventory


def verify_inventory(root: Path = ROOT, path: Path | None = None) -> dict:
    inventory = load_inventory(path or root / "artifacts/inventory-v1.json")
    errors = []
    for entry in inventory["files"]:
        try:
            raw = safe_path(root, entry["path"]).read_bytes()
            if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                errors.append(f"hash/size mismatch: {entry['path']}")
        except (OSError, ValueError) as error:
            errors.append(f"{entry['path']}: {error}")
    return {
        "source_commit": inventory["source_commit"],
        "checked": len(inventory["files"]),
        "ok": not errors,
        "errors": errors,
    }


def load_recipes(root: Path = ROOT) -> dict:
    config = read_json(root / "configs/reproduce.json")
    if config.get("version") != 1 or not isinstance(config.get("recipes"), dict):
        raise ValueError("invalid reproduction configuration")
    for name, recipe in config["recipes"].items():
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError(f"invalid recipe name: {name}")
        if not isinstance(recipe.get("depends"), list):
            raise ValueError(f"invalid dependencies: {name}")
        if not isinstance(recipe.get("argv"), list) or not all(
            isinstance(arg, str) for arg in recipe["argv"]
        ):
            raise ValueError(f"invalid argv: {name}")
        if not isinstance(recipe.get("anchors"), list):
            raise ValueError(f"invalid anchors: {name}")
    topological_order(config["recipes"], list(config["recipes"]))
    topological_order(config["recipes"], config["sample"])
    return config


def topological_order(recipes: dict, selected: list[str]) -> list[str]:
    order, visiting, visited = [], set(), set()

    def visit(name):
        if name not in recipes:
            raise ValueError(f"unknown recipe: {name}")
        if name in visiting:
            raise ValueError(f"dependency cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in recipes[name]["depends"]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in selected:
        visit(name)
    return order


def run_process(argv: list[str], cwd: Path, log: Path, timeout: float) -> None:
    """Own a process group, reap the child and terminate descendants on every exit."""
    env = {
        **os.environ,
        **dict.fromkeys(THREADS, "1"),
        "PYTHONPATH": str(cwd / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
            if status:
                raise RuntimeError(f"child exit {status}; see {log}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def snapshot(root: Path, destination: Path, inventory: dict, frozen_inputs: list[str]) -> None:
    """Copy only immutable inputs; no historical output directories are copied."""
    destination.mkdir()
    names = {
        entry["path"]
        for entry in inventory["files"]
        if entry["path"].startswith(("src/", "scripts/", "configs/", "data/sut/"))
    }
    names.update(("src/prism/cli.py", "src/prism/reproduce.py", "uv.lock"))
    names.update(frozen_inputs)
    for name in sorted(names):
        source = safe_path(root, name)
        target = safe_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())


def record_key(config: dict) -> str:
    """Pair corresponding workload arms even if their full config hashes differ."""
    keys = (
        "task",
        "workload",
        "batch",
        "order",
        "strategy",
        "endpoint",
        "client_concurrency",
        "arrival_rate_per_s",
        "batch_size",
        "policy",
        "queue_capacity",
    )
    return json.dumps({k: config[k] for k in keys if k in config}, sort_keys=True)


def compare_records(root: Path, tree: Path, recipe: dict, threshold: float) -> dict:
    from prism.record import load_run

    anchors = [load_run(safe_path(root, name)) for name in recipe["anchors"]]
    by_key = {}
    for anchor in anchors:
        key = record_key(anchor.config)
        if key in by_key:
            raise ValueError(f"ambiguous historical anchors: {anchor.path}")
        by_key[key] = anchor
    rows, seen = [], set()
    for path in sorted((tree / "runs").rglob("measurement.json")):
        run = load_run(path.parent)
        # A new calibration generates a new fit/service model. It is deliberately
        # excluded from historical equality, even when its workload key matches.
        derived = ("after" in path.parent.name and run.config.get("task") == "P1-T3") or (
            run.config.get("phase") == "P3" and run.config.get("mode") == "模拟"
        )
        key = record_key(run.config)
        anchor = None if derived else by_key.get(key)
        row = {
            "run": str(path.parent.relative_to(tree)),
            "config_hash": run.measurement.config_hash,
            "env_fingerprint": run.measurement.env_fingerprint,
        }
        if anchor is None:
            row["status"] = "new-derived" if derived else "unanchored"
        else:
            if key in seen:
                raise ValueError(f"duplicate replay workload: {key}")
            seen.add(key)
            equal = run.measurement.config_hash == anchor.measurement.config_hash
            env_equal = run.measurement.env_fingerprint == anchor.measurement.env_fingerprint
            simulated = run.config.get("mode") in ("模拟", "解析模型")
            row.update(
                anchor=str(anchor.path.relative_to(root)),
                config_equal=equal,
                env_equal=env_equal,
                status="matched" if equal else "config-mismatch",
            )
            if simulated:
                artifacts = (
                    ["predictions.json"]
                    if run.config["mode"] == "解析模型"
                    else ["summary.json", "requests.json", "events.json", "queue.json"]
                )
                row["deterministic_equal"] = all(
                    read_json(path.parent / name) == read_json(anchor.path / name)
                    for name in artifacts
                )
            else:
                row["drift_ms"] = {
                    q: getattr(run.measurement, q) - getattr(anchor.measurement, q)
                    for q in ("p50", "p95", "p99")
                }
                row["t4_annotation"] = {
                    q: "below-resolution" if abs(delta) < threshold else "above-resolution"
                    for q, delta in row["drift_ms"].items()
                }
            row["historical_env_fingerprint"] = anchor.measurement.env_fingerprint
            row["historical_config_hash"] = anchor.measurement.config_hash
        rows.append(row)
    missing = [str(a.path.relative_to(root)) for k, a in by_key.items() if k not in seen]
    return {
        "records": rows,
        "missing_anchors": missing,
        "ok": not missing
        and all(
            r["status"] != "config-mismatch" and r.get("deterministic_equal", True) for r in rows
        ),
    }


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def reproduce(
    selected: list[str],
    *,
    dry_run: bool = False,
    output: Path | None = None,
    timeout: float = 1800,
    root: Path = ROOT,
) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    config = load_recipes(root)
    order = topological_order(config["recipes"], selected)
    plan = {
        "order": order,
        "recipes": {name: config["recipes"][name] for name in order},
        "notes": config["notes"],
        "dry_run": dry_run,
    }
    if dry_run:
        return plan  # No mkdir, file write, environment collection or process startup.
    verification = verify_inventory(root)
    if not verification["ok"]:
        raise ValueError(f"inventory verification failed: {verification['errors']}")
    base = safe_path(root, "runs/.reproduce")
    if output is not None:
        output = output.absolute()
        if not output.is_relative_to(base.absolute()):
            raise ValueError("-o must be a new file under runs/.reproduce")
        safe_path(root, output.relative_to(root.absolute()).as_posix())
        if output.exists():
            raise FileExistsError(output)
    base.mkdir(parents=True, exist_ok=True)
    session = base / uuid4().hex
    session.mkdir()
    report = {
        "session": str(session),
        "verification": verification,
        "notes": config["notes"],
        "t4": config["t4"],
        "recipes": [],
        "ok": True,
    }
    inventory = load_inventory(root / "artifacts/inventory-v1.json")
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum, frame):
        raise KeyboardInterrupt("reproduction cancelled")

    signal.signal(signal.SIGTERM, cancel)
    try:
        for name in order:
            recipe = config["recipes"][name]
            tree = session / name
            row = {"name": name, "argv": [sys.executable, *recipe["argv"]]}
            report["recipes"].append(row)
            try:
                snapshot(root, tree, inventory, config["frozen_inputs"])
                run_process(row["argv"], tree, session / f"{name}.log", timeout)
                row.update(compare_records(root, tree, recipe, config["t4"]["resolution_ms"]))
                # T4 has only a Markdown historical anchor, not a standard run.
                if name == "calibrate":
                    text = (tree / "results/p0/calibration.md").read_text()
                    historical = (root / "results/p0/calibration.md").read_text()
                    for label, pattern in (
                        ("config", r"config 哈希[^\n]*：`([a-f0-9]{64})`"),
                        ("env", r"env 指纹：`([a-f0-9]{64})`"),
                    ):
                        current = re.findall(pattern, text)
                        original = re.findall(pattern, historical)
                        row[f"report_{label}_hashes"] = current
                        row[f"historical_report_{label}_hashes"] = original
                        row[f"report_{label}_equal"] = bool(current) and current == original
                    row["ok"] = row["ok"] and row["report_config_equal"]
            except (Exception, KeyboardInterrupt) as error:
                row.update(ok=False, error=f"{type(error).__name__}: {error}")
            report["ok"] = report["ok"] and row["ok"]
            write_json_new(session / f"{name}.json", row)
            if not row["ok"]:
                break
    finally:
        signal.signal(signal.SIGTERM, previous)
        write_json_new(session / "report.json", report)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json_new(output, report)
    return report

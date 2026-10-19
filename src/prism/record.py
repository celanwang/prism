"""Persist and summarize complete, validated experiment records."""

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prism import __version__
from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement

_SLUG = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
_RUN_NAME = re.compile(r"[0-9]{8}-(?P<slug>[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)")
_ENV_FIELDS = {
    "platform",
    "python",
    "cpu_count",
    "memory_bytes",
    "packages",
    "lock_hash",
    "prism_version",
    "captured_at",
}


@dataclass
class Run:
    """A validated run and its original provenance objects."""

    path: Path
    slug: str
    config: dict[str, Any]
    measurement: Measurement
    env: dict[str, Any]
    meta: dict[str, Any]


def _json_object(value: str) -> dict[str, Any]:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    result = json.loads(value, object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    json.dumps(result, allow_nan=False)
    return result


def _timestamp(value: Any) -> None:
    if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
        raise ValueError("expected an ISO timestamp with timezone")


def _validate(
    config: dict[str, Any], measurement: Measurement, env: dict[str, Any], meta: dict[str, Any]
) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if measurement.config_hash != config_hash:
        raise ValueError("config hash does not match measurement.config_hash")
    if not _ENV_FIELDS <= env.keys():
        raise ValueError("env.lock is missing environment fields")
    if env_fingerprint(env) != measurement.env_fingerprint:
        raise ValueError("env fingerprint does not match measurement.env_fingerprint")
    if not isinstance(measurement.samples_ms, list) or not measurement.samples_ms:
        raise ValueError("measurement must contain timing samples")
    numbers = [
        *measurement.samples_ms,
        measurement.mean,
        measurement.stdev,
        measurement.p50,
        measurement.p95,
        measurement.p99,
    ]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in numbers
    ):
        raise ValueError("measurement timings must be finite nonnegative numbers")
    for value in (measurement.peak_tracemalloc_bytes, measurement.max_rss_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("measurement memory peaks must be nonnegative integers")
    if not {"created_at", "prism_version", "task", "phase"} <= meta.keys():
        raise ValueError("meta.json is missing required fields")
    if not isinstance(meta["prism_version"], str) or not meta["prism_version"]:
        raise ValueError("meta.json must contain a prism version")
    for value in (measurement.captured_at, env["captured_at"], meta["created_at"]):
        _timestamp(value)


def save_run(
    runs_dir: str | Path, slug: str, config: dict[str, Any], measurement: Measurement
) -> Path:
    """Validate and atomically publish a run; existing destinations are never replaced.

    Directory flock serializes cooperating writers, including the existence check
    and rename, so even an existing empty directory is refused. Staging lives on
    the same filesystem and is cleaned up if writing or publishing fails.
    """
    if not isinstance(slug, str) or _SLUG.fullmatch(slug) is None:
        raise ValueError("slug must use alphanumerics separated by '-' or '_'")
    # Snapshot JSON before hashing/writing, including normalization of JSON keys.
    config = _json_object(json.dumps(config, sort_keys=True, allow_nan=False))
    measurement = Measurement.from_json(measurement.to_json())
    env = collect_env()
    now = datetime.now(UTC)
    meta = {
        "created_at": now.isoformat(),
        "prism_version": __version__,
        "task": config.get("task"),
        "phase": config.get("phase"),
    }
    _validate(config, measurement, env, meta)
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{now:%Y%m%d}-{slug}"
    directory = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        if os.path.lexists(path):
            raise FileExistsError(f"run already exists: {path}")
        with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=root) as staging:
            temporary = Path(staging)
            contents = {
                "config.json": json.dumps(config, sort_keys=True, indent=2, allow_nan=False),
                "measurement.json": measurement.to_json(),
                "env.lock": json.dumps(env, sort_keys=True, indent=2, allow_nan=False),
                "meta.json": json.dumps(meta, sort_keys=True, indent=2, allow_nan=False),
            }
            for name, content in contents.items():
                (temporary / name).write_text(content + "\n", encoding="utf-8")
            temporary.rename(path)
    finally:
        os.close(directory)
    return path


def load_run(path: str | Path) -> Run:
    """Fail closed on missing/malformed files or inconsistent provenance."""
    path = Path(path)
    match = _RUN_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid run directory name: {path.name}")
    try:
        config, measurement_data, env, meta = (
            _json_object((path / name).read_text(encoding="utf-8"))
            for name in ("config.json", "measurement.json", "env.lock", "meta.json")
        )
        measurement = Measurement.from_json(json.dumps(measurement_data))
        _validate(config, measurement, env, meta)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"invalid run {path}: {error}") from error
    return Run(path, match["slug"], config, measurement, env, meta)


def summarize(runs_dir: str | Path) -> str:
    """Return a deterministic Markdown table; reject any invalid visible record."""
    rows = [
        "| slug | p50 (ms) | p95 (ms) | p99 (ms) | peak_tracemalloc_bytes | "
        "max_rss_bytes | env fingerprint | config hash |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    root = Path(runs_dir)
    if not root.exists():
        return "\n".join(rows)
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue  # Unpublished staging directories are never experiment evidence.
        run = load_run(path)
        m = run.measurement
        rows.append(
            f"| {run.slug} | {m.p50:.6g} | {m.p95:.6g} | {m.p99:.6g} | "
            f"{m.peak_tracemalloc_bytes} | {m.max_rss_bytes} | "
            f"{m.env_fingerprint} | {m.config_hash[:8]} |"
        )
    return "\n".join(rows)

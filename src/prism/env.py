"""Collect reproducible experiment environment information."""

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from prism import __version__

_REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_env() -> dict[str, Any]:
    """Describe the current environment, isolating its capture timestamp."""
    memory_bytes = None
    if sys.platform == "darwin":
        memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    lock_path = _REPO_ROOT / "uv.lock"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None
    return {
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0],
        },
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "packages": {name: metadata.version(name) for name in ("numpy", "pytest", "ruff")},
        "lock_hash": lock_hash,
        "prism_version": __version__,
        "captured_at": datetime.now(UTC).isoformat(),
    }


def env_fingerprint(env: dict[str, Any]) -> str:
    """Hash sorted environment JSON, excluding only the capture timestamp."""
    stable_env = {key: value for key, value in env.items() if key != "captured_at"}
    return hashlib.sha256(json.dumps(stable_env, sort_keys=True).encode("utf-8")).hexdigest()

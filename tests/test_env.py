import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pytest

from prism import __version__, cli
from prism import env as env_module
from prism.env import collect_env, env_fingerprint


def test_collect_env_reports_installed_environment() -> None:
    env = collect_env()
    assert set(env) == {
        "platform",
        "python",
        "cpu_count",
        "memory_bytes",
        "packages",
        "lock_hash",
        "prism_version",
        "captured_at",
    }
    assert env["platform"] == {
        "system": platform.system().lower(),
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
    }
    assert env["python"] == sys.version
    assert env["cpu_count"] == os.cpu_count()
    if sys.platform == "darwin":
        assert isinstance(env["memory_bytes"], int)
        assert env["memory_bytes"] > 0
    else:
        assert env["memory_bytes"] is None
    assert env["packages"] == {name: metadata.version(name) for name in ("numpy", "pytest", "ruff")}
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    assert env["lock_hash"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert env["prism_version"] == __version__
    assert datetime.fromisoformat(env["captured_at"]).utcoffset() is not None


def test_lock_hash_uses_repository_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"locked dependencies\n")
    (tmp_path / "uv.lock").write_bytes(b"unrelated lock\n")
    monkeypatch.setattr(env_module, "_REPO_ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert collect_env()["lock_hash"] == hashlib.sha256(b"locked dependencies\n").hexdigest()


def test_missing_lock_is_null(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_module, "_REPO_ROOT", tmp_path)
    assert collect_env()["lock_hash"] is None


def test_fingerprint_excludes_timestamp_and_sorts_nested_keys() -> None:
    first = {"packages": {"ruff": "1", "numpy": "2"}, "captured_at": "first"}
    second = {"captured_at": "second", "packages": {"numpy": "2", "ruff": "1"}}
    expected = hashlib.sha256(b'{"packages": {"numpy": "2", "ruff": "1"}}').hexdigest()
    assert env_fingerprint(first) == env_fingerprint(first) == expected
    assert env_fingerprint(second) == expected
    assert env_fingerprint({"packages": first["packages"]}) == expected
    assert first["captured_at"] == "first"
    assert env_fingerprint({**first, "cpu_count": 8}) != expected


def test_env_cli_prints_json_from_other_directory(tmp_path: Path) -> None:
    outputs = [
        subprocess.run(
            [sys.executable, "-m", "prism", "env"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    environments = [json.loads(output) for output in outputs]
    for output, env in zip(outputs, environments, strict=True):
        assert output == json.dumps(env, sort_keys=True, indent=2) + "\n"
        assert env["lock_hash"] is not None
    assert env_fingerprint(environments[0]) == env_fingerprint(environments[1])


def test_env_write_uses_repository_path_and_prints_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"
    env = collect_env()
    monkeypatch.setattr(cli, "__file__", str(root / "src" / "prism" / "cli.py"))
    monkeypatch.setattr(cli, "collect_env", lambda: env)
    monkeypatch.setattr(sys, "argv", ["prism", "env", "--write"])
    monkeypatch.chdir(tmp_path)
    cli.main()
    path = root / "configs" / "env.lock"
    assert path.read_text(encoding="utf-8") == json.dumps(env, sort_keys=True, indent=2) + "\n"
    assert capsys.readouterr().out == f"{path}\n{env_fingerprint(env)}\n"
    assert not (tmp_path / "configs").exists()

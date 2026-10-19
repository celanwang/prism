"""Record tests use synthetic measurements, never performance assertions."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from prism import __version__
from prism import record as record_module
from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement
from prism.record import load_run, save_run, summarize


@pytest.fixture
def config() -> dict[str, Any]:
    return {"nested": {"b": 2, "a": 1}, "task": "P0-T5", "phase": "P0"}


@pytest.fixture
def measurement(config: dict[str, Any]) -> Measurement:
    return Measurement(
        samples_ms=[1.0],
        mean=1.0,
        stdev=0.0,
        p50=1.0,
        p95=1.0,
        p99=1.0,
        peak_tracemalloc_bytes=10,
        max_rss_bytes=20,
        config_hash=hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        env_fingerprint=env_fingerprint(collect_env()),
        captured_at=datetime.now(UTC).isoformat(),
    )


def test_save_load_roundtrip(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement
) -> None:
    path = save_run(tmp_path / "runs", "round-trip", config, measurement)
    assert path.name.endswith("-round-trip")
    assert len(path.name.split("-", 1)[0]) == 8
    assert {file.name for file in path.iterdir()} == {
        "config.json",
        "measurement.json",
        "env.lock",
        "meta.json",
    }
    run = load_run(path)
    assert run.path == path
    assert run.slug == "round-trip"
    assert run.config == config
    assert run.measurement == measurement
    assert (path / "measurement.json").read_text() == measurement.to_json() + "\n"
    assert env_fingerprint(run.env) == measurement.env_fingerprint
    assert run.meta == {
        "created_at": run.meta["created_at"],
        "prism_version": __version__,
        "task": "P0-T5",
        "phase": "P0",
    }
    assert datetime.fromisoformat(run.meta["created_at"]).utcoffset() is not None


def test_duplicate_slug_does_not_change_existing_run(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement
) -> None:
    path = save_run(tmp_path, "duplicate", config, measurement)
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    with pytest.raises(FileExistsError):
        save_run(tmp_path, "duplicate", config, measurement)
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}
    assert list(tmp_path.iterdir()) == [path]


def test_existing_empty_destination_is_refused(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement
) -> None:
    path = tmp_path / f"{datetime.now(UTC):%Y%m%d}-empty"
    path.mkdir()
    with pytest.raises(FileExistsError):
        save_run(tmp_path, "empty", config, measurement)
    assert list(path.iterdir()) == []


def test_concurrent_writers_publish_exactly_one_run(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement
) -> None:
    def save() -> Path | None:
        try:
            return save_run(tmp_path, "concurrent", config, measurement)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: save(), range(4)))
    assert sum(path is not None for path in paths) == 1
    assert len(list(tmp_path.iterdir())) == 1
    assert load_run(next(tmp_path.iterdir())).measurement == measurement


def test_partial_write_is_never_published_and_staging_is_cleaned(
    tmp_path: Path,
    config: dict[str, Any],
    measurement: Measurement,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.write_text

    def write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == "env.lock":
            assert len(list(tmp_path.iterdir())) == 1
            assert next(tmp_path.iterdir()).name.startswith(".")
            assert len(summarize(tmp_path).splitlines()) == 2
            raise OSError("injected disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="injected disk failure"):
        save_run(tmp_path, "partial", config, measurement)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["config.json", "measurement.json", "env.lock", "meta.json"])
def test_missing_file_is_rejected(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement, filename: str
) -> None:
    path = save_run(tmp_path, "missing", config, measurement)
    (path / filename).rename(path / f".{filename}.missing")
    with pytest.raises(ValueError, match="invalid run"):
        load_run(path)
    with pytest.raises(ValueError, match="invalid run"):
        summarize(tmp_path)


@pytest.mark.parametrize(
    ("filename", "field", "value"),
    [
        ("config.json", "extra", "tampered"),
        ("env.lock", "cpu_count", 999999),
        ("measurement.json", "config_hash", "0" * 64),
        ("measurement.json", "env_fingerprint", "0" * 64),
        ("measurement.json", "p50", "invalid"),
        ("measurement.json", "samples_ms", []),
        ("meta.json", "created_at", "invalid"),
    ],
)
def test_inconsistent_or_malformed_records_fail_closed(
    tmp_path: Path,
    config: dict[str, Any],
    measurement: Measurement,
    filename: str,
    field: str,
    value: Any,
) -> None:
    path = save_run(tmp_path, "tampered", config, measurement)
    target = path / filename
    data = json.loads(target.read_text())
    data[field] = value
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_run(path)
    with pytest.raises(ValueError):
        summarize(tmp_path)


@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"a": 1, "a": 2}', '{"x": NaN}'])
@pytest.mark.parametrize("filename", ["config.json", "measurement.json", "env.lock", "meta.json"])
def test_invalid_json_and_incomplete_objects_fail_closed(
    tmp_path: Path,
    config: dict[str, Any],
    measurement: Measurement,
    filename: str,
    contents: str,
) -> None:
    path = save_run(tmp_path, "invalid-json", config, measurement)
    (path / filename).write_text(contents)
    with pytest.raises(ValueError):
        load_run(path)


@pytest.mark.parametrize("slug", ["", ".", "..", "../escape", "/absolute", "a/b", "a|b", "a\nb"])
def test_invalid_slugs_never_write(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement, slug: str
) -> None:
    with pytest.raises(ValueError, match="slug"):
        save_run(tmp_path, slug, config, measurement)
    assert list(tmp_path.iterdir()) == []


def test_save_rejects_mismatched_config_and_environment_before_writing(
    tmp_path: Path,
    config: dict[str, Any],
    measurement: Measurement,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="config hash"):
        save_run(tmp_path, "config", {**config, "changed": True}, measurement)
    env = collect_env()
    monkeypatch.setattr(record_module, "collect_env", lambda: {**env, "cpu_count": 999999})
    with pytest.raises(ValueError, match="env fingerprint"):
        save_run(tmp_path, "environment", config, measurement)
    assert list(tmp_path.iterdir()) == []


def test_summary_contains_every_run_in_sorted_order(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement
) -> None:
    save_run(tmp_path, "zebra", config, measurement)
    save_run(tmp_path, "alpha", config, measurement)
    table = summarize(tmp_path)
    assert len(table.splitlines()) == 4
    assert table.index("alpha") < table.index("zebra")
    assert measurement.config_hash[:8] in table
    assert measurement.env_fingerprint in table
    assert "peak_tracemalloc_bytes" in table and "max_rss_bytes" in table


def test_empty_and_missing_runs_return_empty_table(tmp_path: Path) -> None:
    table = summarize(tmp_path)
    assert len(table.splitlines()) == 2
    assert "slug" in table and "p50" in table and "p95" in table and "p99" in table
    assert summarize(tmp_path / "does-not-exist") == table
    assert not (tmp_path / "does-not-exist").exists()


@pytest.mark.parametrize("source", ["stdin", "file"])
def test_cli_save_summary_and_duplicate_error(
    tmp_path: Path, config: dict[str, Any], measurement: Measurement, source: str
) -> None:
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    command = [
        sys.executable,
        "-m",
        "prism",
        "record",
        "save",
        "--slug",
        "cli",
        "--config",
        str(config_path),
    ]
    if source == "file":
        measurement_path = tmp_path / "input-measurement.json"
        measurement_path.write_text(measurement.to_json())
        command.extend(["--measurement", str(measurement_path)])
    result = subprocess.run(
        command,
        input=measurement.to_json(),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert load_run(tmp_path / result.stdout.strip()).measurement == measurement
    summary_command = [sys.executable, "-m", "prism", "record", "summary"]
    summary = subprocess.run(
        summary_command, cwd=tmp_path, text=True, capture_output=True, check=True
    )
    assert "| cli |" in summary.stdout
    duplicate = subprocess.run(
        command, input=measurement.to_json(), cwd=tmp_path, text=True, capture_output=True
    )
    assert duplicate.returncode == 1
    assert "already exists" in duplicate.stderr
    assert "Traceback" not in duplicate.stderr
    path = tmp_path / result.stdout.strip()
    (path / "config.json").write_text("{}")
    invalid = subprocess.run(summary_command, cwd=tmp_path, text=True, capture_output=True)
    assert invalid.returncode == 1
    assert invalid.stdout == ""
    assert "config hash" in invalid.stderr


def test_cli_summary_empty_custom_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "prism", "record", "summary", "--runs-dir", "empty"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == summarize(tmp_path / "empty") + "\n"

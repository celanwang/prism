"""Release manifest, failure evidence, comparisons and an actual clean-clone smoke."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prism.reproduce import load_recipes, read_json, topological_order

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/release_check.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_manifest_covers_phase_reports_and_rejects_bad_inputs(tmp_path):
    config = release.load_manifest(ROOT, ROOT / "configs/release.json")
    assert set(config["phases"]) == {f"P{i}" for i in range(6)}
    for change in ("version", "timeout", "missing-phase", "path", "pr", "duplicate-json"):
        value = copy.deepcopy(config)
        if change == "version":
            value["version"] = 2
        elif change == "timeout":
            value["timeout_seconds"] = float("inf")
        elif change == "missing-phase":
            value["phases"].pop("P3")
        elif change == "path":
            value["phases"]["P2"]["reports"] = ["results/p2/../../README.md"]
        elif change == "pr":
            value["phases"]["P1"]["prs"] = [True]
        path = tmp_path / "manifest.json"
        path.write_text(
            '{"version":1,"version":1}' if change == "duplicate-json" else json.dumps(value)
        )
        with pytest.raises(ValueError):
            release.load_manifest(ROOT, path)


def test_readme_commands_all_parsed_and_shell_rejected():
    commands = release.readme_commands((ROOT / "README.md").read_text())
    assert ["uv", "sync", "--locked"] in commands
    assert ["uv", "run", "--locked", "prism", "reproduce", "--sample"] in commands
    for text in ("no commands", "```sh\nrm x\n```", "```sh\nuv run --locked x; env\n```"):
        with pytest.raises(ValueError):
            release.readme_commands(text)


def test_documents_missing_evidence_is_not_a_pass(tmp_path, monkeypatch):
    config = release.load_manifest(ROOT, ROOT / "configs/release.json")
    for name in (
        "docs/ROADMAP.md",
        "artifacts/inventory-v1.json",
        "README.md",
        "AGENTS.md",
        "DESIGN.md",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(release, "git", lambda *args: "Merge pull request #17")
    result = release.audit_documents(tmp_path, config)
    assert not result["ok"]
    assert any("evidence/pointer" in error for error in result["errors"])
    assert any("no local merge" in error for error in result["errors"])


def test_document_audit_current_evidence_and_merged_prs():
    config = release.load_manifest(ROOT, ROOT / "configs/release.json")
    result = release.audit_documents(ROOT, config)
    assert result["ok"], result["errors"]
    assert all(row["inventory_covered"] for row in result["phases"] if row["phase"] != "P5")


def comparison_fixture():
    config = {
        "sample": ["measured"],
        "recipes": {
            "sut": {"depends": [], "anchors": []},
            "measured": {"depends": ["sut"], "anchors": ["history/arm"]},
        },
    }
    row = {
        "anchor": "history/arm",
        "status": "matched",
        "config_equal": True,
        "config_hash": "a" * 64,
        "historical_config_hash": "a" * 64,
        "env_fingerprint": "b" * 64,
        "historical_env_fingerprint": "c" * 64,
        "env_equal": False,
        "drift_ms": {q: 1.0 for q in ("p50", "p95", "p99")},
        "t4_annotation": dict.fromkeys(("p50", "p95", "p99"), "above-resolution"),
    }
    report = {
        "ok": True,
        "recipes": [
            {"name": "sut", "ok": True, "records": [], "missing_anchors": []},
            {"name": "measured", "ok": True, "records": [row], "missing_anchors": []},
        ],
    }
    return config, report


def test_comparison_reports_environment_and_requires_full_coverage():
    config, report = comparison_fixture()
    result = release.review_comparison(report, config, check=False)
    assert result["ok"] and result["recipes"][1]["env_differences"] == 1
    for mutate in (
        lambda r: r["recipes"].pop(),
        lambda r: r["recipes"][1]["records"].clear(),
        lambda r: r["recipes"][1]["records"][0].update(config_equal=False),
        lambda r: r["recipes"][1]["records"][0].update(historical_config_hash="d" * 64),
        lambda r: r["recipes"][1]["records"][0].update(env_equal=True),
        lambda r: r["recipes"][1]["records"][0].pop("env_equal"),
        lambda r: r["recipes"][1]["records"][0].update(deterministic_equal=False),
        lambda r: r["recipes"][1]["records"][0].pop("t4_annotation"),
    ):
        value = copy.deepcopy(report)
        mutate(value)
        assert not release.review_comparison(value, config, check=False)["ok"]


def test_process_failure_preserves_output_and_status(tmp_path):
    runner = release.Runner(tmp_path, 10)
    with pytest.raises(RuntimeError, match="exit 7"):
        runner.run(
            [sys.executable, "-c", "print('failure evidence');raise SystemExit(7)"], tmp_path
        )
    assert "failure evidence" in (tmp_path / "00.log").read_text()
    assert runner.steps[0]["returncode"] == 7
    assert (tmp_path / "00.json").is_file()


def test_failed_release_persists_summary_and_stops(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("candidate source rejected")

    monkeypatch.setattr(release, "prepare_clone", fail)
    output = release.release_check(check=True, runs_dir=tmp_path)
    result = read_json(output / "release.json")
    assert not result["ok"] and result["steps"] == []
    assert "candidate source rejected" in result["error"]


def test_isolation_discards_developer_overrides(monkeypatch):
    for name in (
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "GIT_DIR",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "polluted")
    env = release.clean_env()
    assert not any(value == "polluted" for value in env.values())


def test_check_real_clone_new_venv_and_replay(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release_check.py"),
            "--check",
            "--runs-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    outputs = list(tmp_path.glob("*-p5-release-*-check"))
    assert len(outputs) == 1
    result = read_json(outputs[0] / "release.json")
    assert result["ok"] and result["smoke_only"]
    assert result["source"]["clone_clean_before_sync"]
    assert result["source"]["venv_absent_before_sync"]
    assert result["clone_status_after"] == ""
    report = read_json(outputs[0] / "reproduction.json")
    assert [r["name"] for r in report["recipes"]] == topological_order(
        load_recipes()["recipes"], ["sut"]
    )
    assert (outputs[0] / "reproduction.tar.gz").is_file()
    assert all(s["returncode"] == 0 for s in result["steps"])

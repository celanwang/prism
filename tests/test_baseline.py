"""Verify baseline assembly and evidence provenance, never assert performance."""

import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from prism.env import collect_env, env_fingerprint
from prism.record import load_run
from prism.sut import DEFAULT_SUT_PATH, load_sut

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("baseline_script", ROOT / "scripts" / "baseline.py")
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


@pytest.fixture
def recipe() -> dict:
    config = deepcopy(baseline.load_config(baseline.CONFIG_PATH))
    config["prefill"].update(sequence_lengths=[1, 2, 3], warmup=0, repeat=2)
    config["decode"].update(prompt_ids=[256, 316], max_new_tokens=2, warmup=0, repeat=2)
    return config


def test_repository_recipe_and_load(tmp_path: Path) -> None:
    config = baseline.load_config(baseline.CONFIG_PATH)
    assert config["prefill"]["sequence_lengths"] == [1, 32, 128]
    assert config["decode"]["max_new_tokens"] == 32
    assert config["calibration"]["resolution_ms"] == 0.092584
    assert config["threads"] == 1
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert baseline.load_config(path) == config


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "threads", True),
        (None, "device", "mps"),
        (None, "sut_path", ""),
        ("prefill", "warmup", -1),
        ("prefill", "repeat", 0),
        ("prefill", "repeat", True),
        ("prefill", "sequence_lengths", []),
        ("prefill", "sequence_lengths", [1, 1]),
        ("prefill", "sequence_lengths", [False]),
        ("prefill", "token_id", -1),
        ("decode", "prompt_ids", []),
        ("decode", "prompt_ids", [True]),
        ("decode", "max_new_tokens", 0),
        ("calibration", "resolution_ms", float("nan")),
        ("calibration", "resolution_ms", float("inf")),
        ("calibration", "resolution_ms", 0),
        ("calibration", "report", ""),
    ],
)
def test_invalid_recipe(recipe: dict, section: str | None, key: str, value: object) -> None:
    target = recipe if section is None else recipe[section]
    target[key] = value
    with pytest.raises(ValueError):
        baseline.validate_config(recipe)


@pytest.mark.parametrize("value", [[], {}, {"unsupported": 1}])
def test_missing_recipe_sections(value: object) -> None:
    with pytest.raises(ValueError):
        baseline.validate_config(value)


def test_workload_inputs_follow_recipe(recipe: dict) -> None:
    sut = Mock()
    sut.config.context_length = 128
    sut.config.vocab_size = 512
    workloads = baseline.make_workloads(sut, recipe)
    for workload, length in zip(workloads[:-1], recipe["prefill"]["sequence_lengths"], strict=True):
        workload.fn()
        tokens = sut.forward.call_args.args[0]
        assert tokens.shape == (1, length)
        assert tokens.dtype == np.int64
        assert np.all(tokens == recipe["prefill"]["token_id"])
        assert workload.tokens == length
        assert workload.warmup == recipe["prefill"]["warmup"]
        assert workload.repeat == recipe["prefill"]["repeat"]
    workloads[-1].fn()
    sut.generate.assert_called_once_with(
        recipe["decode"]["prompt_ids"],
        max_new_tokens=recipe["decode"]["max_new_tokens"],
        eos_id=None,
    )
    assert workloads[-1].tokens == recipe["decode"]["max_new_tokens"]


def test_real_decode_has_exact_new_token_count(recipe: dict) -> None:
    sut = load_sut(DEFAULT_SUT_PATH)
    prompt = list(recipe["decode"]["prompt_ids"])
    workload = baseline.make_workloads(sut, recipe)[-1]
    for _ in range(2):
        tokens = workload.fn()
        assert tokens[: len(prompt)] == prompt
        assert len(tokens) == len(prompt) + recipe["decode"]["max_new_tokens"]
    assert recipe["decode"]["prompt_ids"] == prompt


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("prefill", "sequence_lengths", [129]),
        ("prefill", "token_id", 512),
        ("decode", "prompt_ids", [512]),
    ],
)
def test_inputs_outside_model_limits_fail_before_measurement(
    recipe: dict, section: str, key: str, value: object
) -> None:
    recipe[section][key] = value
    sut = Mock()
    sut.config.context_length = 128
    sut.config.vocab_size = 512
    with pytest.raises(ValueError):
        baseline.make_workloads(sut, recipe)
    sut.forward.assert_not_called()
    sut.generate.assert_not_called()


def test_tiny_end_to_end_and_report_provenance(
    tmp_path: Path, recipe: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Isolate thread environment mutations when calling the CLI in-process.
    for name in baseline.THREAD_VARIABLES:
        monkeypatch.setenv(name, "7")
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")
    runs_dir, report_path = tmp_path / "runs", tmp_path / "results" / "REPORT.md"
    args = ["--config", str(config_path), "--runs-dir", str(runs_dir), "--report", str(report_path)]
    baseline.main(args)
    assert capsys.readouterr().out.strip() == str(report_path)
    paths = baseline.matching_runs(runs_dir, recipe)
    assert len(paths) == len(recipe["prefill"]["sequence_lengths"]) + 1
    report = report_path.read_text(encoding="utf-8")
    assert report == baseline.render_report(paths)
    for section in (
        "P0 阶段验收总览",
        "环境与测量配方",
        "基准数字表",
        "记录与配置哈希",
        "T4 分辨率对照",
        "边界声明",
        "复现与验收入口",
    ):
        assert section in report
    assert "O(n²)" in report and "无 KV cache" in report
    assert "0.092584" in report and "不可分辨" in report
    fingerprint = env_fingerprint(collect_env())
    for path in paths:
        assert {file.name for file in path.iterdir()} == {
            "config.json",
            "measurement.json",
            "env.lock",
            "meta.json",
        }
        run = load_run(path)
        m, w = run.measurement, run.config["workload"]
        assert len(m.samples_ms) == recipe[w["kind"]]["repeat"]
        assert run.config["recipe"] == recipe
        assert m.env_fingerprint == fingerprint
        assert m.config_hash in report and fingerprint in report
        assert str(path) in report
        row = (
            f"| {w['name']} | {w['warmup']} | {w['repeat']} | {m.p50:.9f} | {m.p95:.9f} | "
            f"{m.p99:.9f} | {w['tokens'] * 1000 / m.p50:.6f} | "
            f"{m.peak_tracemalloc_bytes} | {m.max_rss_bytes} |"
        )
        assert row in report
    snapshot = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    baseline.main([*args, "--check"])
    assert capsys.readouterr().out == report
    assert snapshot == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    baseline.main([*args, "--from-runs", *map(str, paths)])
    assert capsys.readouterr().out.strip() == str(report_path)
    assert report_path.read_text(encoding="utf-8") == report
    # A second independent measurement batch keeps stable provenance hashes.
    baseline.main(args)
    capsys.readouterr()
    rerun_paths = baseline.matching_runs(runs_dir, recipe)
    assert set(rerun_paths).isdisjoint(paths)
    before = {load_run(path).config["workload"]["name"]: load_run(path) for path in paths}
    for path in rerun_paths:
        run = load_run(path)
        old = before[run.config["workload"]["name"]]
        assert run.measurement.config_hash == old.measurement.config_hash
        assert run.measurement.env_fingerprint == old.measurement.env_fingerprint
    mixed = [rerun_paths[0], *paths[1:]]
    with pytest.raises(ValueError, match="single measurement batch"):
        baseline.render_report(mixed)
    with pytest.raises(ValueError, match="exactly one"):
        baseline.render_report(paths[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        baseline.render_report([*paths, paths[0]])


def test_check_missing_evidence_does_not_write(tmp_path: Path, recipe: dict) -> None:
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="no complete saved baseline"):
        baseline.main(
            [
                "--config",
                str(config_path),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--report",
                str(tmp_path / "REPORT.md"),
                "--check",
            ]
        )
    assert list(tmp_path.iterdir()) == [config_path]


def test_cli_sets_thread_limits_before_numpy_import(tmp_path: Path, recipe: dict) -> None:
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")
    runs_dir = tmp_path / "runs"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "baseline.py"),
        "--config",
        str(config_path),
        "--runs-dir",
        str(runs_dir),
        "--report",
        str(tmp_path / "REPORT.md"),
    ]
    env = {**os.environ, **dict.fromkeys(baseline.THREAD_VARIABLES, "8")}
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    for path in baseline.matching_runs(runs_dir, recipe):
        assert load_run(path).config["runtime"]["thread_environment"] == dict.fromkeys(
            baseline.THREAD_VARIABLES, "1"
        )
    result = subprocess.run([*command, "--check"], check=True, capture_output=True, text=True)
    assert result.stdout == (tmp_path / "REPORT.md").read_text(encoding="utf-8")

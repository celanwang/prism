"""Determinism, T3 gates, bounded client semantics and genuine small HTTP flow."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from prism.record import load_run
from prism.schedsim import simulate

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("serve_load", ROOT / "scripts/serve_load.py")
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.fixture
def recipe():
    return json.loads(script.CONFIG_PATH.read_text())


def test_seeded_load_exactly_matches_t3_arrivals_and_lengths(recipe):
    calibrated = script.validate_recipe(recipe)
    config = script.simulation_config(recipe, calibrated, 40, 64)
    workload = script.make_workload(config, 512)
    assert workload == script.make_workload(config, 512)
    assert workload != script.make_workload({**config, "seed": config["seed"] + 1}, 512)
    oracle = simulate(config)
    assert len(workload) == len(oracle.requests)
    for request, reference in zip(workload, oracle.requests, strict=True):
        assert request["arrival_ms"] == reference.arrival_ms
        assert len(request["prompt"]) == reference.prompt_tokens
        assert request["max_new_tokens"] == reference.output_tokens
    for capacity in recipe["queue_capacities"]:
        assert workload == script.make_workload({**config, "queue_capacity": capacity}, 512)


@pytest.mark.parametrize("rate", [10, 40, 160])
@pytest.mark.parametrize("capacity", [2, 64])
def test_serial_adapter_and_rule_match_original_t3(recipe, rate, capacity):
    config = script.simulation_config(recipe, script.validate_recipe(recipe), rate, capacity)
    workload = script.make_workload(config, 512)
    expected = simulate(config)
    observed = script.serial_simulation(workload, config, None)["summary"]
    for key in ("completed", "rejected", "avalanche", "avalanche_reasons"):
        assert observed[key] == expected.summary[key]
    for key in ("p50", "p95", "p99"):
        assert observed["latency_ms"][key] == pytest.approx(expected.summary["latency_ms"][key])
    for key in ("late_window_mean_queues", "queue_slope_per_s", "mean_queue_length"):
        assert observed[key] == pytest.approx(expected.summary[key])
    # Independent T3 event trace checks the integration and the strict OR predicate.
    gate = script.avalanche(expected.summary["latency_ms"], expected.queue_series, config)
    for key in ("late_window_mean_queues", "queue_slope_per_s", "mean_queue_length"):
        assert gate[key] == pytest.approx(expected.summary[key])
    assert gate["avalanche_reasons"] == expected.summary["avalanche_reasons"]


def test_avalanche_or_strict_boundaries_and_final_half_weighting(recipe):
    config = {
        **recipe,
        "duration_ms": 8000,
        "queue_growth": {"min_queue": 20, "min_slope_per_s": 1},
    }
    increasing = [
        {"time_ms": t, "queue_length": q}
        for t, q in [(0, 0), (4000, 1), (5000, 5), (6000, 10), (7000, 20)]
    ]
    result = script.avalanche({"p99": 1000}, increasing, config)
    assert result["avalanche_reasons"] == ["finite_window_queue_growth_proxy"]
    assert result["late_window_mean_queues"] == [1, 5, 10, 20]
    flat = [{"time_ms": 0, "queue_length": 0}]
    assert not script.avalanche({"p99": 1000}, flat, config)["avalanche"]
    assert script.avalanche({"p99": 1000.001}, flat, config)["avalanche_reasons"] == [
        "p99_threshold"
    ]
    assert not script.avalanche({"p99": None}, flat, config)["avalanche"]
    equal = deepcopy(increasing)
    equal[3]["queue_length"] = 5
    assert not script.avalanche({"p99": None}, equal, config)["avalanche"]
    too_short = deepcopy(increasing)
    too_short[-1]["queue_length"] = 19
    assert not script.avalanche({"p99": None}, too_short, config)["avalanche"]
    boundary = {**config, "queue_growth": {"min_queue": 20, "min_slope_per_s": 19 / 3}}
    assert not script.avalanche({"p99": None}, increasing, boundary)["avalanche"]


def test_client_full_and_queue_reject_are_separate_and_timeout_drains(recipe):
    config = script.simulation_config(recipe, script.validate_recipe(recipe), 160, 2)
    workload = [
        {"id": i, "arrival_ms": float(i), "prompt": [42] * 16, "max_new_tokens": 64}
        for i in range(10)
    ]
    bounded = script.serial_simulation(workload, config, 1)["summary"]
    assert bounded["completed"] == 1
    assert bounded["client_full"] == 9
    assert bounded["rejected"] == 0
    queued = script.serial_simulation(workload, config, 10)["summary"]
    assert queued["completed"] == 3
    assert queued["rejected"] == 7
    timeout = script.serial_simulation(workload, {**config, "client_timeout_s": 0.001}, 1)
    assert timeout["summary"]["timed_out"] == 3
    assert timeout["summary"]["completed"] == 0
    assert timeout["summary"]["latency_ms"]["p99"] is None
    assert max(r["finish_ms"] for r in timeout["requests"]) > 9


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("device", "mps"),
        ("native_threads", 2),
        ("concurrency", [1, 1]),
        ("concurrency", [True]),
        ("queue_capacities", [0]),
        ("repeat", 0),
        ("duration_ms", float("inf")),
        ("http_workers", 8),
        ("client_timeout_s", 0.1),
        ("calibrated_recipe_sha256", "bad"),
    ],
)
def test_bad_recipe_fails_before_artifacts(recipe, key, value, tmp_path):
    recipe[key] = value
    with pytest.raises(ValueError):
        script.run_load(recipe, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_small_real_http_grid_and_persistence(recipe, tmp_path, capsys):
    original = deepcopy(recipe)
    recipe["check"].update(
        arrival_rates_per_s=[160], concurrency=[1, 8], queue_capacities=[2], duration_ms=100
    )
    result = script.run_load(recipe, tmp_path / "runs", check=True)
    root = Path(result["path"])
    assert len(result["rows"]) == 2
    assert result["warmup"] == 0 and result["repeat"] == 2
    assert "非实验证据" in result["evidence"]
    assert (
        hashlib.sha256((root / "calibrated_schedsim.json").read_bytes()).hexdigest()
        == (recipe["calibrated_recipe_sha256"])
    )
    workloads = []
    for row in result["rows"]:
        run = load_run(root / row["point"])
        assert len(run.measurement.samples_ms) == 2
        assert run.measurement.config_hash == row["config_hash"]
        rounds = json.loads((run.path / "rounds.json").read_text())
        workloads.append(json.loads((run.path / "workload.json").read_text()))
        for trial in rounds:
            summary = trial["summary"]
            assert summary["arrivals"] == sum(
                summary[k]
                for k in (
                    "completed",
                    "rejected",
                    "timed_out",
                    "client_full",
                    "http_503",
                    "transport_errors",
                )
            )
            assert summary["http_503"] == 0
            assert summary["transport_errors"] == 0
            for request in trial["requests"]:
                if request["status"] == 200:
                    assert request["e2e_ms"] >= request["wire_ms"]
                    assert request["queue_wait_ms"] >= 0
            assert "client_cpu_ms" in summary and "dispatch_ms" in summary
            assert "max_sampling_gap_ms" in summary
    assert workloads[0] == workloads[1]
    assert original["concurrency"] == recipe["concurrency"]
    assert "## 雪崩点实测 vs 模拟" in (root / "REPORT.md").read_text()
    capsys.readouterr()


def test_stage_report_sections_and_boundaries(recipe):
    script.validate_recipe(recipe)
    report = (ROOT / "results/p3/REPORT.md").read_text()
    for section in (
        "部署等价性",
        "并发–尾时延曲线",
        "背压行为",
        "雪崩点实测 vs 模拟",
        "偏差分析",
        "边界声明",
        "复现与验收",
    ):
        assert f"## {section}" in report
    for term in ("#12", "单进程串行推理", "loopback", "无 TLS", "config", "env", "客户端"):
        assert term in report

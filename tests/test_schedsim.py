"""Verify 模拟 event semantics and accounting, never absolute performance claims."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from prism.record import load_run
from prism.schedsim import (
    config_hash,
    decode_ms,
    sample_length,
    save_simulation,
    simulate,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("schedsim_script", ROOT / "scripts" / "schedsim.py")
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


@pytest.fixture
def recipe() -> dict:
    return json.loads(script.CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def config(recipe: dict) -> dict:
    point = script.grid_points(recipe)[0]
    point.update(
        duration_ms=1000,
        arrival_rate_per_s=50,
        policy="continuous",
        batch_size=4,
        batch_timeout_ms=0,
        prompt_length={"kind": "uniform", "min": 2, "max": 2},
        output_length={"kind": "uniform", "min": 3, "max": 3},
        max_batched_tokens=128,
    )
    point["service_model"].update(prefill_ms_per_token=1, decode_nodes=[[1, 1], [4, 2.5]])
    return point


def arrivals(monkeypatch: pytest.MonkeyPatch, times: list[float]) -> None:
    """Inject exact interarrival times; constant length distributions need no RNG."""
    gaps = np.diff([0, *times, 1e9]).tolist()
    monkeypatch.setattr(
        "prism.schedsim.np.random.default_rng",
        lambda seed: Mock(exponential=Mock(side_effect=gaps)),
    )


@pytest.mark.parametrize("policy", ["static", "continuous"])
def test_same_seed_every_event_and_samples(config: dict, policy: str) -> None:
    config["policy"] = policy
    config["prompt_length"].update(max=12)
    config["output_length"] = {"kind": "truncated_normal", "min": 1, "max": 20, "mean": 5, "std": 5}
    original = deepcopy(config)
    first, second = simulate(config), simulate(config)
    assert first.events and first == second
    assert config == original
    assert [event["time_ms"] for event in first.events] == sorted(
        event["time_ms"] for event in first.events
    )
    config["seed"] += 1
    assert first.events != simulate(config).events


@pytest.mark.parametrize("policy", ["static", "continuous"])
def test_littles_law_low_load(config: dict, policy: str) -> None:
    config.update(policy=policy, duration_ms=1_000_000, arrival_rate_per_s=5, batch_timeout_ms=2)
    result = simulate(config)
    summary = result.summary
    # L = lambda W for the system; Lq = lambda Wq for the waiting queue.
    # Long, low-load window makes nominal-vs-empirical arrival and end effects small.
    rate = config["arrival_rate_per_s"] / 1000
    assert summary["mean_in_system"] == pytest.approx(rate * summary["latency_ms"]["mean"], rel=0.1)
    assert summary["mean_queue_length"] == pytest.approx(
        rate * summary["wait_ms"]["mean"], rel=0.15, abs=0.0001
    )


def test_extremely_low_load_policies_equal_without_batch_wait(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrivals(monkeypatch, [1, 101, 201])
    continuous = simulate(config)
    arrivals(monkeypatch, [1, 101, 201])
    config["policy"] = "static"
    static = simulate(config)
    assert static.requests == continuous.requests
    assert static.events == continuous.events
    assert static.summary["latency_ms"] == continuous.summary["latency_ms"]


def test_static_timeout_is_expected_low_load_difference(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.update(policy="static", batch_timeout_ms=5)
    arrivals(monkeypatch, [1, 101])
    result = simulate(config)
    assert [request.wait_ms for request in result.requests] == [5, 5]


def test_static_departs_full_before_timeout(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    config.update(policy="static", batch_size=2, batch_timeout_ms=100)
    arrivals(monkeypatch, [1, 2])
    result = simulate(config)
    assert [request.start_ms for request in result.requests] == [2, 2]
    assert result.requests[0].finish_ms == pytest.approx(
        2 + 4 + 3 * decode_ms(config["service_model"], 2)
    )


def test_continuous_admits_only_at_iteration_boundary(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrivals(monkeypatch, [1, 1.5, 3.5])
    result = simulate(config)
    # First request: prefill [1,3], decode [3,4]; arrivals during both wait until 4.
    assert [request.start_ms for request in result.requests] == [1, 4, 4]
    assert result.requests[0].finish_ms > 1 + 2 + 3  # inserted prefill blocks old decode
    starts = {event["time_ms"] for event in result.events if event["kind"] == "iteration"}
    assert all(request.start_ms in starts for request in result.requests)


def test_static_does_not_refill_and_continuous_does(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrivals(monkeypatch, [1, 1.5])
    continuous = simulate(config)
    arrivals(monkeypatch, [1, 1.5])
    config["policy"] = "static"
    static = simulate(config)
    assert static.requests[1].start_ms == static.requests[0].finish_ms
    assert continuous.requests[1].start_ms < continuous.requests[0].finish_ms


def test_static_retains_padded_width(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    config.update(policy="static", batch_timeout_ms=10, batch_size=2)
    # Draw varying outputs deterministically using the ordinary length sampler.
    rng = Mock()
    rng.exponential.side_effect = [1, 1, 1e9]
    rng.integers.side_effect = [1, 3]
    monkeypatch.setattr("prism.schedsim.np.random.default_rng", lambda seed: rng)
    config["output_length"].update(min=1, max=3)
    result = simulate(config)
    steps = [event for event in result.events if event["kind"] == "decode_start"]
    assert [len(event["ids"]) for event in steps] == [2, 1, 1]
    assert [event["width"] for event in steps] == [2, 2, 2]


@pytest.mark.parametrize("policy", ["static", "continuous"])
def test_token_budget_and_no_starvation(
    config: dict, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    config.update(policy=policy, max_batched_tokens=4, batch_size=4)
    arrivals(monkeypatch, [1, 1.1, 1.2, 1.3, 1.4])
    result = simulate(config)
    assert result.summary["completed"] == 5
    for event in result.events:
        if event["kind"] == "iteration":
            assert event["batched_tokens"] == len(event["ids"]) + event["prefill_tokens"]
            assert event["batched_tokens"] <= 4
            assert len(event["ids"]) <= 4


def test_tie_completion_timeout_arrival_order(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.update(batch_size=1, queue_timeout_ms=4)
    arrivals(monkeypatch, [1, 2, 6])
    result = simulate(config)
    kinds = [event["kind"] for event in result.events if event["time_ms"] == 6]
    assert kinds[:3] == ["decode_end", "timeout", "arrival"]
    assert result.requests[1].status == "timed_out"
    assert result.requests[2].start_ms == 6


def test_rejections_timeouts_and_terminal_accounting(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.update(batch_size=1, queue_capacity=1, queue_timeout_ms=1)
    arrivals(monkeypatch, [1, 1.1, 1.2, 10])
    result = simulate(config)
    assert [request.status for request in result.requests] == [
        "completed",
        "timed_out",
        "rejected",
        "completed",
    ]
    summary = result.summary
    assert (summary["completed"], summary["timed_out"], summary["rejected"]) == (2, 1, 1)
    assert (
        sum(summary[key] for key in ("completed", "timed_out", "rejected")) == summary["arrivals"]
    )
    assert all(item["queue_length"] <= 1 and item["in_system"] >= 0 for item in result.queue_series)
    assert result.queue_series[-1]["in_system"] == 0


def test_latency_threshold_and_queue_growth_avalanche(config: dict) -> None:
    config.update(arrival_rate_per_s=1000, duration_ms=4000, batch_size=1, p99_threshold_ms=1)
    result = simulate(config)
    assert result.summary["avalanche"]
    assert "p99_threshold" in result.summary["avalanche_reasons"]
    assert "finite_window_queue_growth_proxy" in result.summary["avalanche_reasons"]
    config["p99_threshold_ms"] = 1e9
    assert simulate(config).summary["avalanche_reasons"] == ["finite_window_queue_growth_proxy"]


def test_drain_and_window_accounting(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    config["duration_ms"] = 5
    arrivals(monkeypatch, [1, 4])
    result = simulate(config)
    summary = result.summary
    assert summary["completed"] == 2
    assert all(request.finish_ms > 5 for request in result.requests)
    assert summary["drained_until_ms"] == max(request.finish_ms for request in result.requests)
    assert summary["completed_output_tokens"] == 6
    # At t=4 the second request is admitted; its prefill blocks the next decode.
    assert summary["window_output_tokens"] == 1
    assert summary["throughput_tokens_per_s"] == 1 / 0.005
    # No request has left before the observation horizon: exact area under N(t).
    assert summary["mean_in_system"] == pytest.approx(((5 - 1) + (5 - 4)) / 5)


def test_empty_window_and_save_refuses_invented_samples(
    config: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arrivals(monkeypatch, [2000])
    result = simulate(config)
    assert result.summary["arrivals"] == 0
    assert result.summary["latency_ms"]["p99"] is None
    assert result.summary["throughput_tokens_per_s"] == 0
    with pytest.raises(ValueError, match="without completed requests"):
        save_simulation(result, tmp_path / "runs", "empty")
    assert not (tmp_path / "runs").exists()


def test_distributions_and_piecewise_curve(config: dict) -> None:
    rng = np.random.default_rng(42)
    for kind in ("uniform", "truncated_normal"):
        spec = {"kind": kind, "min": 2, "max": 9, "mean": 5, "std": 2}
        values = [sample_length(rng, spec) for _ in range(500)]
        assert all(type(value) is int and 2 <= value <= 9 for value in values)
        assert len(set(values)) > 1
    assert decode_ms(config["service_model"], 2) == 1.5
    with pytest.raises(ValueError):
        decode_ms(config["service_model"], 5)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mode", "measurement"),
        ("policy", "unknown"),
        ("seed", True),
        ("seed", -1),
        ("batch_size", 0),
        ("batch_size", 5),
        ("duration_ms", float("inf")),
        ("arrival_rate_per_s", float("nan")),
        ("arrival_rate_per_s", 0),
        ("max_batched_tokens", 2),
        ("batch_timeout_ms", -1),
        ("queue_capacity", 0),
        ("queue_timeout_ms", 0),
        ("p99_threshold_ms", False),
        ("prompt_length", {"kind": "uniform", "min": 2, "max": 1}),
        ("output_length", {"kind": "truncated_normal", "min": 1, "max": 3, "mean": 2, "std": 0}),
        ("output_length", {"kind": "truncated_normal", "min": 1, "max": 3, "mean": 1e99, "std": 1}),
        ("queue_growth", {"min_queue": 0, "min_slope_per_s": 1}),
    ],
)
def test_invalid_config(config: dict, key: str, value: object) -> None:
    config[key] = value
    with pytest.raises(ValueError):
        simulate(config)


@pytest.mark.parametrize(
    "nodes", [[], [[2, 1], [4, 2]], [[1, 1], [1, 2]], [[1, 2], [4, 1]], [[1, 0], [4, 1]], [[1, 1]]]
)
def test_invalid_decode_nodes(config: dict, nodes: list) -> None:
    config["service_model"]["decode_nodes"] = nodes
    with pytest.raises(ValueError):
        validate_config(config)


def test_saved_records_samples_and_hash(config: dict, tmp_path: Path) -> None:
    result = simulate(config)
    path = save_simulation(result, tmp_path, "schedsim-test")
    run = load_run(path)
    assert run.config == config
    assert run.measurement.config_hash == config_hash(config) == result.summary["config_hash"]
    assert run.measurement.samples_ms == [request.latency_ms for request in result.requests]
    assert run.measurement.p99 == result.summary["latency_ms"]["p99"]
    assert json.loads((path / "summary.json").read_text()) == result.summary
    assert json.loads((path / "events.json").read_text())["events"] == result.events
    assert json.loads((path / "queue.json").read_text())["queue_series"] == result.queue_series
    requests = json.loads((path / "requests.json").read_text())
    assert requests["mode"] == "模拟" and len(requests["requests"]) == len(result.requests)
    with pytest.raises(FileExistsError):
        save_simulation(result, tmp_path, "schedsim-test")


def test_cli_grid_writes_and_check_does_not_write(
    recipe: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    recipe.update(arrival_rates_per_s=[100], batch_sizes=[1, 4], duration_ms=200)
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")
    runs_dir = tmp_path / "runs"
    args = ["--config", str(config_path), "--runs-dir", str(runs_dir)]
    script.main([*args, "--check"])
    output = capsys.readouterr().out
    assert "模拟" in output and "不写入" in output
    assert not runs_dir.exists()
    script.main(args)
    output = capsys.readouterr().out
    paths = list(runs_dir.iterdir())
    assert len(paths) == 4
    for path in paths:
        run = load_run(path)
        assert run.measurement.config_hash in output
        assert str(path) in output
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in runs_dir.rglob("*")
        if path.is_file()
    }
    script.main([*args, "--check"])
    capsys.readouterr()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in runs_dir.rglob("*")
        if path.is_file()
    }


def test_grid_validates_every_point_before_writing(recipe: dict, tmp_path: Path) -> None:
    recipe["batch_sizes"] = [1, 100]
    with pytest.raises(ValueError):
        script.run_grid(recipe, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()

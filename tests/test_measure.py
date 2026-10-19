import hashlib
import json
import math
import resource
import tracemalloc
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from prism import measure as measure_module
from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement, measure


@pytest.mark.parametrize("warmup", [0, 3])
def test_warmup_and_sample_count(warmup: int) -> None:
    calls = 0

    def fn() -> int:
        nonlocal calls
        calls += 1
        return calls

    result = measure(fn, warmup=warmup, repeat=4, config={})
    assert calls == warmup + 4
    assert len(result.samples_ms) == 4
    assert all(isinstance(sample, float) and sample >= 0 for sample in result.samples_ms)
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize("parameter", ["warmup", "repeat"])
@pytest.mark.parametrize("value", [-1, True, False, 1.0, "1", None])
def test_invalid_parameters_fail_before_calling_fn(parameter: str, value: Any) -> None:
    def fn() -> None:
        pytest.fail("invalid parameters must fail before calling fn")

    kwargs = {"warmup": 0, "repeat": 1}
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        measure(fn, **kwargs, config={})


def test_zero_repeat_is_invalid() -> None:
    with pytest.raises(ValueError, match="repeat"):
        measure(lambda: pytest.fail("fn must not run"), warmup=0, repeat=0, config={})


def test_single_sample_has_zero_stdev() -> None:
    result = measure(lambda: None, warmup=0, repeat=1, config={})
    assert result.stdev == 0.0
    assert result.mean == result.p50 == result.p95 == result.p99 == result.samples_ms[0]


def test_known_samples_use_linear_percentiles_and_sample_stdev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(
        [0, 1_000_000, 10_000_000, 12_000_000, 20_000_000, 24_000_000, 30_000_000, 38_000_000]
    )
    monkeypatch.setattr(measure_module.time, "perf_counter_ns", lambda: next(ticks))
    result = measure(lambda: None, warmup=2, repeat=4, config={})
    assert result.samples_ms == [1.0, 2.0, 4.0, 8.0]
    assert result.mean == 3.75
    assert result.stdev == pytest.approx(math.sqrt(115 / 12))
    assert result.p50 == 3.0
    assert result.p95 == 7.399999999999999
    assert result.p99 == 7.879999999999999


def test_json_roundtrip_and_provenance() -> None:
    config = {"z": 1, "nested": {"b": 2, "a": "value"}}
    first = measure(lambda: None, warmup=1, repeat=2, config=config)
    second = measure(
        lambda: None,
        warmup=1,
        repeat=2,
        config={"nested": {"a": "value", "b": 2}, "z": 1},
    )
    expected_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert first.config_hash == second.config_hash == expected_hash
    assert first.env_fingerprint == second.env_fingerprint == env_fingerprint(collect_env())
    assert datetime.fromisoformat(first.captured_at).utcoffset() is not None
    assert first.to_json() == json.dumps(asdict(first), sort_keys=True, indent=2)
    assert Measurement.from_json(first.to_json()) == first
    assert first.peak_tracemalloc_bytes >= 0


def test_config_is_hashed_before_warmup() -> None:
    config = {"calls": 0}
    expected_hash = hashlib.sha256(b'{"calls": 0}').hexdigest()

    def fn() -> None:
        config["calls"] += 1

    result = measure(fn, warmup=2, repeat=1, config=config)
    assert config["calls"] == 3
    assert result.config_hash == expected_hash


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_config_is_rejected_before_warmup(value: float) -> None:
    with pytest.raises(ValueError):
        measure(
            lambda: pytest.fail("invalid config must fail before calling fn"),
            warmup=2,
            repeat=1,
            config={"nested": [value]},
        )
    assert not tracemalloc.is_tracing()


def test_existing_tracing_is_rejected_without_calling_fn_or_stopping_tracing() -> None:
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="exclusive"):
            measure(
                lambda: pytest.fail("existing tracing must fail before calling fn"),
                warmup=2,
                repeat=1,
                config={},
            )
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("warmup", [0, 1])
def test_fn_exception_propagates_and_tracing_is_stopped(warmup: int) -> None:
    error = RuntimeError("fn failed")

    def fn() -> None:
        raise error

    with pytest.raises(RuntimeError) as caught:
        measure(fn, warmup=warmup, repeat=2, config={})
    assert caught.value is error
    assert not tracemalloc.is_tracing()


def test_memory_and_environment_capture_follow_timed_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    tracing_states: list[bool] = []
    events: list[str] = []

    def fn() -> None:
        nonlocal calls
        calls += 1
        tracing_states.append(tracemalloc.is_tracing())
        events.append("fn")

    def clock() -> int:
        events.append("clock")
        return len(events) * 1_000_000

    def traced_memory() -> tuple[int, int]:
        assert calls == 3
        assert tracemalloc.is_tracing()
        events.append("peak")
        return 100, 200

    def getrusage(who: int) -> SimpleNamespace:
        assert who == resource.RUSAGE_SELF
        assert calls == 3
        events.append("rss")
        return SimpleNamespace(ru_maxrss=12345)

    def collect() -> dict[str, Any]:
        assert not tracemalloc.is_tracing()
        events.append("env")
        return {"cpu_count": 8, "captured_at": "excluded"}

    monkeypatch.setattr(measure_module.time, "perf_counter_ns", clock)
    monkeypatch.setattr(measure_module.tracemalloc, "get_traced_memory", traced_memory)
    monkeypatch.setattr(measure_module.resource, "getrusage", getrusage)
    monkeypatch.setattr(measure_module, "collect_env", collect)
    result = measure(fn, warmup=1, repeat=2, config={})
    assert tracing_states == [False, True, True]
    assert events == ["fn", "clock", "fn", "clock", "clock", "fn", "clock", "peak", "rss", "env"]
    assert result.peak_tracemalloc_bytes == 200
    assert result.max_rss_bytes == 12345
    assert result.env_fingerprint == env_fingerprint({"cpu_count": 8})

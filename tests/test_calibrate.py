import hashlib
import importlib.util
import json
import sys
import tracemalloc
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("calibrate_script", ROOT / "scripts" / "calibrate.py")
assert SPEC is not None and SPEC.loader is not None
calibrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibrate
SPEC.loader.exec_module(calibrate)


@pytest.fixture
def config() -> dict:
    recipe = json.loads((ROOT / "configs" / "calibrate.json").read_text(encoding="utf-8"))
    recipe.update(warmup=2, repeat=4)
    recipe["probes"]["compute"]["elements"] = 8
    recipe["probes"]["allocation"].update(objects=3, bytes_per_object=16)
    return recipe


@pytest.fixture
def measurement(config: dict) -> Measurement:
    return Measurement(
        samples_ms=[2.0] * config["repeat"],
        mean=2.0,
        stdev=0.0,
        p50=2.0,
        p95=2.0,
        p99=2.0,
        peak_tracemalloc_bytes=100,
        max_rss_bytes=200,
        config_hash=hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        env_fingerprint=env_fingerprint(collect_env()),
        captured_at="2026-10-19T00:00:00+00:00",
    )


def test_probes_use_configured_parameters_and_reuse_compute_storage(config: dict) -> None:
    assert calibrate.make_probe("noop", {})() is None
    params = {"elements": 5, "initial": 4.0, "increment": 2.0}
    compute = calibrate.make_probe("compute", params)
    first = compute()
    np.testing.assert_array_equal(first, np.full(5, 6.0))
    pointer = first.ctypes.data
    second = compute()
    assert first is second and second.ctypes.data == pointer
    np.testing.assert_array_equal(second, np.full(5, 8.0))
    fresh = calibrate.make_probe("compute", params)()
    assert not np.shares_memory(first, fresh)
    np.testing.assert_array_equal(fresh, np.full(5, 6.0))

    allocate = calibrate.make_probe("allocation", config["probes"]["allocation"])
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        objects = allocate()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert after > before
    assert len(objects) == 3
    assert all(isinstance(obj, bytearray) and len(obj) == 16 for obj in objects)
    another = allocate()
    assert len({id(obj) for obj in objects + another}) == 6


def test_compute_passes_preallocated_output(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = calibrate.make_probe("compute", {"elements": 3, "initial": 0, "increment": 2})
    add = Mock(side_effect=lambda values, increment, *, out: out)
    monkeypatch.setattr(calibrate.np, "add", add)
    first, second = probe(), probe()
    assert first is second
    for call in add.call_args_list:
        assert call.args[0] is first
        assert call.args[1] == 2
        assert call.kwargs["out"] is first


def test_bare_timing_uses_direct_clock_and_linear_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0, 1_000_000, 10_000_000, 12_000_000, 20_000_000, 24_000_000])
    states = []
    monkeypatch.setattr(calibrate.time, "perf_counter_ns", lambda: next(ticks))
    forbidden = Mock(side_effect=AssertionError("bare timing must not call measure"))
    monkeypatch.setattr(calibrate, "measure", forbidden)
    timing = calibrate.bare_timing(
        lambda: states.append(tracemalloc.is_tracing()), warmup=2, repeat=3
    )
    assert states == [False] * 5
    assert timing.samples_ms == [1.0, 2.0, 4.0]
    assert timing.mean == pytest.approx(7 / 3)
    assert (timing.p50, timing.p95, timing.p99) == pytest.approx((2, 3.8, 3.96))
    forbidden.assert_not_called()


def test_bare_rejects_existing_tracing_without_stopping_it() -> None:
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="inactive"):
            calibrate.bare_timing(lambda: pytest.fail("must not run"), warmup=0, repeat=1)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_measure_receives_normalized_recipe_for_each_probe(
    config: dict, measurement: Measurement, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = deepcopy(config)
    bare = Mock(return_value=calibrate.Timing([1.0] * 4, 1.0, 1.0, 1.0, 1.0))
    traced = Mock(return_value=measurement)
    monkeypatch.setattr(calibrate, "bare_timing", bare)
    monkeypatch.setattr(calibrate, "measure", traced)
    results = calibrate.calibrate(config)
    assert [result.name for result in results] == ["noop", "compute", "allocation"]
    assert config == snapshot
    assert bare.call_count == traced.call_count == 3
    for bare_call, traced_call in zip(bare.call_args_list, traced.call_args_list, strict=True):
        assert callable(traced_call.args[0])
        assert bare_call.args[0] is not traced_call.args[0]
        assert bare_call.kwargs == {"warmup": 2, "repeat": 4}
        assert traced_call.kwargs == {"warmup": 2, "repeat": 4, "config": config}


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "warmup", -1),
        (None, "warmup", True),
        (None, "repeat", 0),
        (None, "repeat", 1.5),
        ("compute", "elements", 0),
        ("compute", "initial", float("nan")),
        ("compute", "increment", float("inf")),
        ("allocation", "objects", False),
        ("allocation", "bytes_per_object", -2),
    ],
)
def test_invalid_recipe_fails_before_measurement(
    config: dict, section: str | None, key: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = config if section is None else config["probes"][section]
    target[key] = value
    forbidden = Mock(side_effect=AssertionError("invalid recipe must not be measured"))
    monkeypatch.setattr(calibrate, "measure", forbidden)
    with pytest.raises(ValueError):
        calibrate.calibrate(config)
    forbidden.assert_not_called()


def test_report_fields_signed_differences_and_empirical_threshold(
    config: dict, measurement: Measurement
) -> None:
    bare = calibrate.Timing([1.0] * 4, 1.0, 1.0, 1.0, 1.0)
    comparisons = [calibrate.Comparison(name, bare, measurement) for name in config["probes"]]
    report = calibrate.render_report(config, comparisons, collect_env(), "test CPU")
    for field in (
        measurement.config_hash,
        measurement.env_fingerprint,
        "test CPU",
        "p50",
        "p95",
        "p99",
        "mean",
        "可分辨最小收益",
        "低于 1.000000 ms 的差异不作为证据",
        "预热 2 次，重复 4 次",
        "置信区间",
        "放大倍数",
    ):
        assert field in report
    for result in comparisons:
        assert result.differences() == dict.fromkeys(calibrate.STATISTICS, 1.0)
        assert result.threshold_ms() == 1.0
        for key in calibrate.STATISTICS:
            assert f"| {result.name} | {key} | 1.000000000 | 2.000000000 | +1.000000000" in report
    measurement.mean = measurement.p50 = measurement.p95 = measurement.p99 = 0.5
    assert comparisons[0].differences() == dict.fromkeys(calibrate.STATISTICS, -0.5)
    assert comparisons[0].threshold_ms() == 0.5
    measurement.p99 = 4.0
    assert comparisons[0].threshold_ms() == 3.5  # Tail spread dominates signed differences.
    measurement.env_fingerprint = "different environment"
    with pytest.raises(RuntimeError, match="provenance"):
        calibrate.render_report(config, comparisons, collect_env(), "test CPU")


def test_cli_check_is_read_only_and_write_overwrites(
    config: dict,
    measurement: Measurement,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(config), encoding="utf-8")
    report_path = tmp_path / "new-directory" / "report.md"
    bare = calibrate.Timing([1.0] * 4, 1.0, 1.0, 1.0, 1.0)
    results = [calibrate.Comparison(name, bare, measurement) for name in config["probes"]]
    monkeypatch.setattr(calibrate, "REPORT_PATH", report_path)
    monkeypatch.setattr(calibrate, "calibrate", lambda config: results)
    monkeypatch.setattr(calibrate, "cpu_model", lambda: "test CPU")
    env = collect_env()
    monkeypatch.setattr(calibrate, "collect_env", lambda: env)
    monkeypatch.chdir(tmp_path)
    calibrate.main(["--config", str(recipe_path), "--check"])
    printed = capsys.readouterr().out
    assert "可分辨最小收益" in printed
    assert not report_path.parent.exists()
    for _ in range(2):
        calibrate.main(["--config", str(recipe_path)])
        assert report_path.read_text(encoding="utf-8") == printed
        assert capsys.readouterr().out.strip() == str(report_path)
    before = report_path.stat().st_mtime_ns
    calibrate.main(["--config", str(recipe_path), "--check"])
    assert capsys.readouterr().out == printed
    assert report_path.stat().st_mtime_ns == before


def test_repository_recipe_and_real_measurement_smoke(config: dict) -> None:
    calibrate.validate_config(json.loads(calibrate.CONFIG_PATH.read_text(encoding="utf-8")))
    results = calibrate.calibrate(config)
    assert len(results) == 3
    for result in results:
        assert len(result.bare.samples_ms) == len(result.traced.samples_ms) == config["repeat"]
        assert not tracemalloc.is_tracing()

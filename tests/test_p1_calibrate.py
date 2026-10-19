"""Check calibration semantics, persistence and report coverage without speed assertions."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prism.record import load_run
from prism.schedsim import decode_ms

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_calibrate", ROOT / "scripts/p1_calibrate.py")
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.fixture
def recipe() -> dict:
    return json.loads(script.CONFIG_PATH.read_text(encoding="utf-8"))


def test_signed_relative_deviation_and_ols_intercept() -> None:
    assert script.relative_deviation(3, 2) == 0.5
    assert script.relative_deviation(1, 2) == -0.5
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            script.relative_deviation(1, invalid)
    points = [{"size": t, "p50": 2 + 0.25 * t} for t in (1, 32, 128)]
    fit = script.fit_prefill(points)
    assert fit["slope_ms_per_token"] == pytest.approx(0.25)
    assert fit["intercept_ms"] == pytest.approx(2)
    assert fit["residuals_ms"] == pytest.approx([0, 0, 0], abs=1e-12)
    with pytest.raises(ValueError, match="slope"):
        script.fit_prefill([{"size": t, "p50": 1000 - t} for t in (1, 32, 128)])


def test_proxy_has_independent_fixed_caches_and_exactly_b_steps(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def forward(sut, tokens, past=None):
        calls.append((tokens.copy(), past))
        return np.zeros((1, tokens.shape[1], 2)), object()

    monkeypatch.setattr("prism.kvcache._forward_cached", forward)
    sut = SimpleNamespace(config=SimpleNamespace(vocab_size=512))
    step = script.make_decode_proxy(sut, 4, 32, 42)
    assert len(calls) == 4
    assert all(tokens.shape == (1, 32) and past is None for tokens, past in calls)
    calls.clear()
    step()
    first = list(calls)
    assert len(first) == 4
    assert len({id(past) for _, past in first}) == 4
    assert all(tokens.shape == (1, 1) for tokens, _ in first)
    calls.clear()
    step()
    assert len(calls) == 4
    assert all(a[1] is b[1] for a, b in zip(first, calls, strict=True))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("decode", "batch_sizes", [1, 4, 8]),
        ("prefill", "sequence_lengths", [1, 128]),
        ("decode", "repeat", 0),
        ("prefill", "warmup", True),
        ("decode", "context_tokens", 0),
        (None, "threads", 2),
        (None, "device", "mps"),
        ("calibration", "resolution_ms", float("nan")),
    ],
)
def test_bad_recipe_fails_before_any_artifact(recipe, section, key, value, tmp_path):
    (recipe if section is None else recipe[section])[key] = value
    with pytest.raises(ValueError):
        script.run_calibration(recipe, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_small_real_flow_and_exact_simulation_replay(recipe, tmp_path, monkeypatch, capsys):
    # Tiny genuine CPU measurement flow; use deterministic timings solely for model
    # assembly so noisy two-sample timings cannot accidentally produce unordered nodes.
    from prism.measure import measure

    seen = []

    def controlled_measure(fn, *, warmup, repeat, config):
        seen.append((warmup, repeat, config))
        result = measure(fn, warmup=warmup, repeat=repeat, config=config)
        size = config["workload"]["size"]
        value = 0.2 + size * 0.01 if config["workload"]["kind"] == "prefill" else size * 0.5
        result.samples_ms = [value] * repeat
        result.mean = result.p50 = result.p95 = result.p99 = value
        result.stdev = 0
        return result

    monkeypatch.setattr("prism.measure.measure", controlled_measure)
    before_recipe = deepcopy(recipe)
    original_path = ROOT / recipe["sim_recipe"]
    original_bytes = original_path.read_bytes()
    runs = tmp_path / "runs"
    result = script.run_calibration(recipe, runs, check=True)
    assert recipe == before_recipe
    assert original_path.read_bytes() == original_bytes
    assert len(seen) == 7
    assert all((warmup, repeat) == (0, 2) for warmup, repeat, _ in seen)
    assert len(list(runs.iterdir())) == 11
    for kind in ("prefill", "decode"):
        for row in result["measurements"][kind]:
            run = load_run(row["path"])
            assert len(run.measurement.samples_ms) == 2
            assert run.measurement.config_hash == row["config_hash"]
            assert "非实验证据" in run.config["evidence"]
    calibrated = json.loads(Path(result["calibrated_recipe"]).read_text())
    original = json.loads(Path(result["original_recipe"]).read_text())
    assert calibrated["service_model"]["decode_nodes"] == [[1, 0.5], [2, 1], [4, 2], [8, 4]]
    assert {k: v for k, v in calibrated.items() if k != "service_model"} == {
        k: v for k, v in original.items() if k != "service_model"
    }
    for row in result["deviations"]:
        assert row["before_relative_deviation"] == pytest.approx(
            (row["before_ms"] - row["measured_p50_ms"]) / row["measured_p50_ms"]
        )
        if row["kind"] == "decode":
            assert row["after_relative_deviation"] == 0
            assert decode_ms(calibrated["service_model"], row["size"]) == row["measured_p50_ms"]
    assert len(result["simulations"]) == 4
    for row in result["simulations"]:
        run = load_run(row["path"])
        assert run.config["arrival_rate_per_s"] == 100
        assert run.config["batch_size"] == 4
        assert run.config["mode"] == "模拟"
    summary = Path(result["calibrated_recipe"]).with_name("calibration.json")
    script.main(["--replay", str(summary), "--runs-dir", str(runs)])
    assert len(list(runs.iterdir())) == 15
    capsys.readouterr()


def test_report_required_sections_and_repository_recipe(recipe):
    script.validate_config(recipe)
    text = (ROOT / "results/p1/REPORT.md").read_text(encoding="utf-8")
    for section in (
        "交付物与证据",
        "正确性门槛",
        "加速比与内存",
        "分页内存–吞吐",
        "模拟–实测校准",
        "偏差分析",
        "边界声明",
        "复现与验收",
    ):
        assert f"## {section}" in text
    for term in ("#6", "#7", "#9", "代理", "0.35", "雪崩", "CPU", "batch-1", "小模型"):
        assert term in text

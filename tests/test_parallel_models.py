"""Verify 解析模型 formulas, byte dimensions, conservation and saved provenance."""

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from prism.parallel_models import ModelSpec, Precision, Strategy, pipeline_bubble, predict
from prism.record import load_run
from prism.sut import DEFAULT_SUT_PATH, load_sut

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parallel_scan", ROOT / "scripts/parallel_scan.py")
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


@pytest.fixture
def model() -> ModelSpec:
    return ModelSpec(layers=4, hidden=16, ffn=64, vocab=128, seq=8, heads=4, position_capacity=16)


@pytest.fixture
def recipe() -> dict[str, Any]:
    return json.loads(script.CONFIG_PATH.read_text(encoding="utf-8"))


def test_sut_parameter_formula_matches_checkpoint(recipe: dict[str, Any]) -> None:
    """Validate all parameter terms against actual independent checkpoint shapes."""
    sut = load_sut(DEFAULT_SUT_PATH)
    spec = ModelSpec(**recipe["models"][0]["spec"])
    assert (spec.layers, spec.hidden, spec.vocab, spec.heads, spec.position_capacity) == (
        sut.config.layers,
        sut.config.width,
        sut.config.vocab_size,
        sut.config.heads,
        sut.config.context_length,
    )
    assert spec.ffn == 4 * sut.config.width
    assert spec.parameters == sum(array.size for array in sut.arrays.values()) == 478720
    hypothetical = recipe["models"][1]
    assert "假想" in hypothetical["provenance"]
    assert 6e9 < ModelSpec(**hypothetical["spec"]).parameters < 8e9


@pytest.mark.parametrize(
    "zero,divisors", [(0, (1, 1, 1)), (1, (1, 1, 4)), (2, (1, 4, 4)), (3, (4, 4, 4))]
)
def test_zero_resident_byte_formulas(
    model: ModelSpec, zero: int, divisors: tuple[int, int, int]
) -> None:
    """Counts times bytes/scalar give bytes for each ZeRO stage, including TP/PP."""
    result = predict(model, Strategy(dp=4, tp=2, pp=2, zero=zero))
    memory = result["memory_bytes"]
    for key, width, divisor in zip(
        ("parameters", "gradients", "optimizer"), (2, 2, 12), divisors, strict=True
    ):
        assert memory[key] == model.parameters / 4 * width / divisor
    assert memory["total"] == memory["model_states"] + memory["activations"]
    if zero == 2:
        assert memory["model_states"] == model.parameters / 4 * (2 + (2 + 12) / 4)


@pytest.mark.parametrize("dp", [1, 2, 4, 8])
@pytest.mark.parametrize("recompute", [False, True])
def test_zero3_cluster_total_conservation_fixed_global_batch(
    model: ModelSpec, dp: int, recompute: bool
) -> None:
    """ZeRO cannot create memory: same global microbatch, TP=PP=1, m=1."""
    baseline = predict(model, Strategy(micro_batch=8, recompute=recompute))["memory_bytes"]
    sharded = predict(model, Strategy(dp=dp, zero=3, micro_batch=8 // dp, recompute=recompute))[
        "memory_bytes"
    ]
    for key in ("parameters", "gradients", "optimizer", "activations", "total"):
        assert dp * sharded[key] == pytest.approx(baseline[key])


def test_zero3_does_not_shard_activations(model: ModelSpec) -> None:
    base = predict(model, Strategy())["memory_bytes"]
    zero = predict(model, Strategy(dp=8, zero=3))["memory_bytes"]
    assert zero["activations"] == base["activations"]
    assert 8 * zero["total"] > base["total"]
    # With TP and PP too, model states still conserve; retained activations need not.
    mixed = predict(model, Strategy(dp=4, tp=2, pp=2, zero=3))
    assert 16 * mixed["memory_bytes"]["model_states"] == base["model_states"]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_megatron_activation_bytes_and_replication(model: ModelSpec, tp: int) -> None:
    out = predict(model, Strategy(tp=tp, micro_batch=2))
    # Published half-precision formula: 10 replicated + 24/t sharded + attention.
    expected = (
        model.layers
        * model.seq
        * 2
        * model.hidden
        * (10 + 24 / tp + 5 * model.heads * model.seq / (model.hidden * tp))
    )
    assert out["memory_bytes"]["activations"] == expected
    assert out["activation_replicated_bytes"] == model.layers * model.seq * 2 * model.hidden * 10
    assert out["activation_replicated_bytes"] + out["activation_sharded_bytes"] == expected


@pytest.mark.parametrize("pp,m", [(1, 1), (2, 1), (2, 8), (4, 2), (4, 16)])
def test_1f1b_activation_residency(model: ModelSpec, pp: int, m: int) -> None:
    base = predict(model, Strategy())["memory_bytes"]["activations"]
    out = predict(model, Strategy(pp=pp, micro_batches=m))
    assert out["memory_bytes"]["activations"] == base / pp * min(m, pp)


def test_general_ffn_and_mask_byte_dimensions(model: ModelSpec) -> None:
    model = replace(model, ffn=48)
    out = predict(model, Strategy(tp=2, micro_batch=3), Precision(3, 4, 16, 4, 2))
    # Independent operation inventory, in scalar counts times byte widths.
    sb = model.seq * 3
    replicated = sb * model.hidden * (4 * 4 + 2 * 2)
    sharded = sb * (4 * 4 * model.hidden + 2 * 4 * 48 + (2 * 4 + 2) * 4 * model.seq) / 2
    assert out["memory_bytes"]["activations"] == model.layers * (replicated + sharded)


def test_all_byte_widths_scale_memory_and_communication(model: ModelSpec) -> None:
    st = Strategy(dp=4, tp=2, pp=2, zero=3, micro_batch=2, micro_batches=8)
    precision = Precision()
    first = predict(model, st, precision)
    doubled = predict(model, st, Precision(**{k: 2 * v for k, v in asdict(precision).items()}))
    for key, value in first["memory_bytes"].items():
        assert doubled["memory_bytes"][key] == 2 * value
    for key, value in first["communication_send_bytes_per_step"].items():
        expected = [2 * v for v in value] if isinstance(value, list) else 2 * value
        assert doubled["communication_send_bytes_per_step"][key] == expected
    assert doubled["compute_only_step_s"] == first["compute_only_step_s"]


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_dp_communication_send_bytes(model: ModelSpec, zero: int) -> None:
    out = predict(model, Strategy(dp=4, zero=zero, micro_batches=3), Precision(2, 4, 12, 2, 1))
    comm = out["communication_send_bytes_per_step"]
    q = model.parameters
    assert comm["dp_all_reduce"] == (2 * 3 / 4 * q * 4 if zero == 0 else 0)
    assert comm["reduce_scatter"] == (3 / 4 * q * 4 * (3 if zero >= 2 else 1) if zero else 0)
    assert comm["all_gather"] == (3 / 4 * q * 2 * (6 if zero == 3 else 1) if zero else 0)
    assert comm["p2p"] == comm["tp_all_reduce"] == 0
    assert comm["total"] == comm["all_reduce"] + comm["reduce_scatter"] + comm["all_gather"]


def test_zero_paper_communication_ratios_for_single_microbatch(model: ModelSpec) -> None:
    totals = [
        predict(model, Strategy(dp=8, zero=z))["communication_send_bytes_per_step"]["total"]
        for z in range(4)
    ]
    assert totals[0] == totals[1] == totals[2]
    assert totals[3] == 1.5 * totals[0]


@pytest.mark.parametrize("recompute,collectives", [(False, 4), (True, 6)])
def test_tp_collective_count_and_payload_bytes(
    model: ModelSpec, recompute: bool, collectives: int
) -> None:
    result = predict(
        model, Strategy(tp=4, pp=2, micro_batch=3, micro_batches=8, recompute=recompute)
    )
    # Ring sends 2*(t-1)/t of a full hidden activation per collective.
    expected = collectives * 2 * 8 * 2 * 3 / 4 * (8 * 3 * 16 * 2)
    assert result["communication_send_bytes_per_step"]["tp_all_reduce"] == expected
    if recompute:
        assert result["memory_bytes"]["activations"] == 2 * 2 * 8 * 3 * 16 * 2
        assert result["activation_sharded_bytes"] == 0


@pytest.mark.parametrize("pp,neighbors", [(1, [0]), (2, [1, 1]), (4, [1, 2, 2, 1])])
def test_pp_boundary_send_conservation(model: ModelSpec, pp: int, neighbors: list[int]) -> None:
    out = predict(model, Strategy(pp=pp, micro_batch=2, micro_batches=3))
    comm = out["communication_send_bytes_per_step"]
    payload = 8 * 2 * 16 * 2
    assert comm["p2p_by_stage"] == [n * 3 * payload for n in neighbors]
    assert pp * comm["p2p"] == 2 * (pp - 1) * 3 * payload


@pytest.mark.parametrize("pp", [1, 2, 4, 16])
def test_pp_bubble_elapsed_time_definition_and_limit(pp: int) -> None:
    stage_s = 0.125
    m = 7
    idle_s, useful_s = (pp - 1) * stage_s, m * stage_s
    assert pipeline_bubble(pp, m) == idle_s / (idle_s + useful_s)
    if pp > 1:
        assert pipeline_bubble(pp, 1) > pipeline_bubble(pp, 16) > pipeline_bubble(pp, 10**12)
    assert pipeline_bubble(pp, 10**12) < 2e-11
    assert pipeline_bubble(1, m) == 0


def test_static_flops_seconds_tokens_dimensions(model: ModelSpec) -> None:
    out = predict(
        model, Strategy(dp=2, tp=2, pp=2, micro_batch=3, micro_batches=8), compute_flops_per_s=2e12
    )
    tokens = 2 * 3 * 8 * model.seq
    flops = 6 * model.parameters * tokens + 12 * model.layers * tokens * model.seq * model.hidden
    assert out["global_tokens_per_step"] == tokens
    assert out["training_flops_per_step"] == flops
    seconds = flops / (8 * 2e12) * 9 / 8
    assert out["compute_only_step_s"] == pytest.approx(seconds)
    assert out["compute_only_tokens_per_s"] == pytest.approx(tokens / seconds)
    faster = predict(model, Strategy(), compute_flops_per_s=2e12)
    slower = predict(model, Strategy(), compute_flops_per_s=1e12)
    assert faster["compute_only_tokens_per_s"] == 2 * slower["compute_only_tokens_per_s"]
    recompute = predict(model, Strategy(recompute=True), compute_flops_per_s=1e12)
    assert recompute["training_flops_per_step"] == pytest.approx(
        4 / 3 * slower["training_flops_per_step"]
    )


@pytest.mark.parametrize("field", list(ModelSpec.__dataclass_fields__))
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_model_dimensions(model: ModelSpec, field: str, bad: Any) -> None:
    with pytest.raises(ValueError, match="解析模型"):
        replace(model, **{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"dp": 0},
        {"tp": True},
        {"pp": -1},
        {"micro_batch": 0},
        {"micro_batches": 1.5},
        {"zero": 4},
        {"zero": True},
        {"recompute": 1},
    ],
)
def test_invalid_strategy(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="解析模型"):
        Strategy(**changes)


@pytest.mark.parametrize("field", list(Precision.__dataclass_fields__))
def test_invalid_precision(field: str) -> None:
    with pytest.raises(ValueError, match="解析模型"):
        Precision(**{field: 0})


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan")])
def test_invalid_compute_rate(model: ModelSpec, bad: float) -> None:
    with pytest.raises(ValueError, match="解析模型"):
        predict(model, Strategy(), compute_flops_per_s=bad)


def test_invalid_divisibility_and_capacity(model: ModelSpec) -> None:
    with pytest.raises(ValueError):
        replace(model, heads=3)
    with pytest.raises(ValueError):
        replace(model, seq=17)
    for strategy in (Strategy(tp=3), Strategy(pp=3)):
        with pytest.raises(ValueError):
            predict(model, strategy)
    for pp, m in ((0, 1), (1, 0), (True, 1), (1, 0.5)):
        with pytest.raises(ValueError):
            pipeline_bubble(pp, m)


def test_full_grid_determinism_and_labels(recipe: dict[str, Any]) -> None:
    original = deepcopy(recipe)
    rows = script.scan(recipe)
    assert len(rows) == 2 * 8 * 2 * 2 * 2 == 128
    assert script.scan(recipe) == rows
    assert recipe == original
    assert all(r["mode"] == r["prediction"]["mode"] == "解析模型" for r in rows)
    table = script.render_table(recipe, rows)
    assert "解析模型" in table and "非实测" in table and "不是峰值显存" in table
    assert script.config_hash(recipe) != script.config_hash({**recipe, "changed": True})


def test_save_run_roundtrip(recipe: dict[str, Any], tmp_path: Path) -> None:
    rows = script.scan(recipe)
    path = script.save_scan(recipe, rows, tmp_path)
    run = load_run(path)
    assert "-p4-models-" in path.name
    assert run.config["mode"] == "解析模型"
    assert run.config["measurement_reference_row"] == 0
    assert run.measurement.config_hash == script.config_hash(run.config)
    assert run.measurement.p50 == rows[0]["prediction"]["compute_only_step_s"] * 1000
    assert run.measurement.p50 == run.measurement.p95 == run.measurement.p99
    assert run.measurement.peak_tracemalloc_bytes == run.measurement.max_rss_bytes == 0
    artifact = json.loads((path / "predictions.json").read_text())
    assert artifact["mode"] == "解析模型"
    assert artifact["config_hash"] == run.measurement.config_hash
    assert artifact["rows"] == rows
    assert (path / "scan.md").read_text() == script.render_table(recipe, rows)


def test_check_is_write_free(
    recipe: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script.main(["--check", "--runs-dir", str(tmp_path / "not-created")])
    assert not (tmp_path / "not-created").exists()
    assert "解析模型: --check passed, 128 grid points" in capsys.readouterr().out


@pytest.mark.parametrize(
    "axis", ["models", "strategies", "micro_batch_sizes", "micro_batch_counts", "recompute"]
)
def test_empty_grid_rejected(recipe: dict[str, Any], axis: str) -> None:
    recipe[axis] = []
    with pytest.raises(ValueError, match="解析模型"):
        script.scan(recipe)


def test_invalid_late_point_writes_nothing(recipe: dict[str, Any], tmp_path: Path) -> None:
    recipe["strategies"][-1]["tp"] = 3
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="解析模型"):
        script.main(["--config", str(config), "--runs-dir", str(tmp_path / "not-created")])
    assert not (tmp_path / "not-created").exists()


def test_cli_saves_and_writes_o_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "table.md"
    script.main(["--runs-dir", str(tmp_path / "runs"), "-o", str(output)])
    saved = list((tmp_path / "runs").iterdir())
    assert len(saved) == 1
    assert output.read_text() == (saved[0] / "scan.md").read_text()
    assert "解析模型: save_run artifacts:" in capsys.readouterr().out


def test_check_disallows_o_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        script.main(["--check", "-o", str(tmp_path / "no.md")])
    assert not (tmp_path / "no.md").exists()

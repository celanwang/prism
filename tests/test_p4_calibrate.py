"""Training correctness and calibration semantics; no wall-clock speed assertions."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prism.record import load_run
from prism.sut import DecoderConfig, load_sut

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p4_calibrate", ROOT / "scripts/p4_calibrate.py")
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.fixture
def recipe():
    return json.loads(script.CONFIG_PATH.read_text(encoding="utf-8"))


def tiny_trainer(recipe):
    """Independent tiny, double precision fixture for numerical gradient checking."""
    rng = np.random.default_rng(17)
    shapes = {
        "token_embedding.weight": (7, 4),
        "position_embedding.weight": (4, 4),
        "final_norm.weight": (4,),
        "final_norm.bias": (4,),
    }
    for layer in range(2):
        for name, out, width in (
            ("ln1", 4, None),
            ("ln2", 4, None),
            ("attention.qkv", 12, 4),
            ("attention.output", 4, 4),
            ("up", 16, 4),
            ("down", 4, 16),
        ):
            shapes[f"blocks.{layer}.{name}.weight"] = (out, width) if width else (out,)
            shapes[f"blocks.{layer}.{name}.bias"] = (out,)
    arrays = {key: rng.normal(0, 0.12, shape) for key, shape in shapes.items()}
    for key in arrays:
        if "norm.weight" in key or ".ln1.weight" in key or ".ln2.weight" in key:
            arrays[key] += 1
    sut = SimpleNamespace(config=DecoderConfig(7, 4, 4, 2, 2, 0), arrays=arrays)
    return script.Trainer(sut, recipe["optimizer"])


def test_backward_numerical_gradients_every_parameter_and_tied_embedding(recipe):
    trainer = tiny_trainer(recipe)
    # Repeated tokens check embedding scatter-add and tied output contributions.
    tokens = np.array([[1, 1, 3], [2, 1, 2]], dtype=np.int64)
    targets = np.array([[1, 3, 2], [1, 2, 5]], dtype=np.int64)
    logits, saved = trainer.forward(tokens)
    loss, grad = trainer.cross_entropy(logits, targets)
    trainer.backward(grad, saved, tokens)
    assert np.isfinite(loss)
    analytic = {key: value.copy() for key, value in trainer.grads.items()}
    rng = np.random.default_rng(19)
    for key, parameter in trainer.params.items():
        direction = rng.normal(size=parameter.shape)
        direction /= np.linalg.norm(direction)
        original = parameter.copy()
        epsilon = 1e-5
        parameter[:] = original + epsilon * direction
        plus = trainer.cross_entropy(trainer.forward(tokens)[0], targets)[0]
        parameter[:] = original - epsilon * direction
        minus = trainer.cross_entropy(trainer.forward(tokens)[0], targets)[0]
        parameter[:] = original
        numeric = (plus - minus) / (2 * epsilon)
        expected = float(np.sum(analytic[key] * direction))
        assert expected == pytest.approx(numeric, abs=2e-8, rel=2e-5), key
    # Calling backward twice must zero the old gradients, not propagate them again.
    trainer.backward(grad, saved, tokens)
    for key in analytic:
        np.testing.assert_allclose(trainer.grads[key], analytic[key], rtol=1e-12, atol=1e-12)


def test_real_sut_forward_and_training_update(recipe):
    sut = load_sut(ROOT / recipe["sut_path"])
    trainer = script.Trainer(sut, recipe["optimizer"])
    tokens = np.array([[1, 20, 31, 42], [5, 5, 7, 9]], dtype=np.int64)
    before = {key: value.copy() for key, value in trainer.params.items()}
    np.testing.assert_allclose(
        trainer.forward(tokens)[0], sut.forward(tokens), atol=2e-5, rtol=2e-5
    )
    inventory = trainer.step(tokens, np.roll(tokens, -1, axis=1), capture=True)
    assert trainer.steps == 1 and np.isfinite(trainer.last_loss)
    assert script.inventory_bytes(inventory["layers"]) > 0
    for key in before:
        np.testing.assert_array_equal(sut.arrays[key], before[key])
        assert np.isfinite(trainer.grads[key]).all()
    for key in ("token_embedding.weight", "blocks.0.attention.qkv.weight", "blocks.1.up.weight"):
        assert not np.array_equal(trainer.params[key], before[key])


def test_adamw_bias_correction_and_decoupled_decay(recipe):
    trainer = tiny_trainer(recipe)
    trainer.optimizer.update(learning_rate=0.1, beta1=0.5, beta2=0.75, weight_decay=0.1)
    for value in trainer.params.values():
        value.fill(1)
    for value in trainer.grads.values():
        value.fill(2)
    trainer.update()
    for value in trainer.params.values():
        np.testing.assert_allclose(value, 0.89, atol=1e-9, rtol=0)
    for value in trainer.grads.values():
        value.fill(-1)
    trainer.update()
    for key, value in trainer.params.items():
        np.testing.assert_allclose(value, 0.8811, atol=1e-9, rtol=0)
        np.testing.assert_allclose(trainer.first[key], 0)
        np.testing.assert_allclose(trainer.second[key], 1)


def test_buffer_inventory_counts_backing_allocations_once():
    owner = np.zeros((2, 12), dtype=np.float32)
    arrays = {"q": owner[:, :4], "k": owner[:, 4:8], "v": owner[:, 8:]}
    result = script.storage_inventory(arrays)
    assert len(result) == 1 and result[0]["names"] == ["q", "k", "v"]
    assert script.inventory_bytes(result) == owner.nbytes == 96


def test_absolute_relative_deviation_and_zero_semantics():
    result = script.deviation(3, 2)
    assert result == {
        "predicted_bytes": 3,
        "measured_bytes": 2,
        "signed_delta_bytes": 1,
        "absolute_deviation_bytes": 1,
        "relative_deviation": 0.5,
    }
    result = script.deviation(1, 2)
    assert result["absolute_deviation_bytes"] == 1 and result["relative_deviation"] == -0.5
    assert script.deviation(0, 0)["relative_deviation"] == 0
    assert script.deviation(1, 0)["relative_deviation"] is None
    for a, b in ((-1, 2), (1, -1), (float("inf"), 1), (1, float("nan"))):
        with pytest.raises(ValueError):
            script.deviation(a, b)


def linearity_rows(values):
    return [
        {
            "micro_batch": b,
            "measured_bytes": {"activations": y},
            "prediction": {"memory_bytes": {"activations": b * 120}},
        }
        for b, y in zip((1, 2, 4), values, strict=True)
    ]


def test_linearity_detects_slope_error_intercept_and_nonlinearity():
    fit = script.fit_linearity(linearity_rows([100, 200, 400]))
    assert fit["passes_proportionality_1e_minus_6"]
    assert fit["slope_bytes_per_micro_batch"] == pytest.approx(100)
    assert fit["prediction_minus_measured_slope_bytes"] == pytest.approx(20)
    assert fit["intercept_bytes"] == pytest.approx(0, abs=1e-10)
    # A perfect free-intercept line alone is insufficient to establish B-proportional memory.
    fit = script.fit_linearity(linearity_rows([110, 210, 410]))
    assert fit["r_squared"] == pytest.approx(1)
    assert fit["intercept_bytes"] == pytest.approx(10)
    assert not fit["passes_proportionality_1e_minus_6"]
    assert not script.fit_linearity(linearity_rows([100, 210, 450]))[
        "passes_proportionality_1e_minus_6"
    ]
    with pytest.raises(ValueError):
        script.fit_linearity(linearity_rows([0, 0, 0]))


@pytest.mark.parametrize(
    "key,value",
    [
        ("micro_batch_sizes", [1, 4]),
        ("micro_batch_sizes", [True, 2, 4]),
        ("warmup", True),
        ("repeat", 0),
        ("threads", 2),
        ("device", "cuda"),
        ("sequence_length", 129),
        ("hypothetical_compute_flops_per_s", float("nan")),
    ],
)
def test_invalid_config_writes_nothing(recipe, key, value, tmp_path):
    recipe[key] = value
    with pytest.raises(ValueError):
        script.run_calibration(recipe, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_small_flow_real_measurements_and_run_roundtrip(recipe, tmp_path):
    original = deepcopy(recipe)
    # Extreme small sequence, but the real SUT model and all three batch sizes.
    recipe.update(sequence_length=2, warmup=1, repeat=2)
    result = script.run_calibration(recipe, tmp_path / "runs")
    assert recipe == {**original, "sequence_length": 2, "warmup": 1, "repeat": 2}
    assert result["linearity"]["passes_proportionality_1e_minus_6"]
    assert len(list((tmp_path / "runs").iterdir())) == 3
    for row in result["rows"]:
        run = load_run(row["path"])
        assert row["optimizer_steps"] == 4
        assert len(run.measurement.samples_ms) == 2
        assert row["measurement"] == json.loads(run.measurement.to_json())
        assert run.measurement.config_hash == row["config_hash"] == script.digest(run.config)
        assert run.measurement.peak_tracemalloc_bytes > 0
        assert row["prediction"]["parameters"] == 478720
        assert row["prediction"]["communication_send_bytes_per_step"]["total"] == 0
        for key in ("parameters", "gradients", "optimizer"):
            assert row["deviations"][key]["absolute_deviation_bytes"] == 0
        assert (
            row["measured_bytes"]["all_forward_retained_including_logits"]
            > row["measured_bytes"]["activations"]
        )
        point = json.loads((Path(row["path"]) / "calibration_point.json").read_text())
        assert point == row
    first = Path(result["rows"][0]["path"])
    assert json.loads((first / "calibration.json").read_text()) == result
    assert (first / "calibration.md").read_text() == script.render_calibration(result)
    # Sorted save_run JSON must render identically after loading in another process.
    reloaded = json.loads((first / "calibration.json").read_text())
    assert script.render_calibration(reloaded) == script.render_calibration(result)


def test_check_is_smoke_and_output_cli(recipe, tmp_path, capsys):
    script.main(["--check", "--runs-dir", str(tmp_path / "smoke")])
    paths = list((tmp_path / "smoke").iterdir())
    assert len(paths) == 3
    for path in paths:
        run = load_run(path)
        assert "非实验证据" in run.config["evidence"]
        assert run.config["recipe"]["sequence_length"] == 4
        assert run.config["recipe"]["warmup"] == 0
        assert len(run.measurement.samples_ms) == 2
    recipe.update(sequence_length=1, warmup=0, repeat=1)
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(recipe))
    output = tmp_path / "output.md"
    script.main(
        ["--config", str(config), "--runs-dir", str(tmp_path / "evidence"), "-o", str(output)]
    )
    assert "# P4-T2 校准数据" in output.read_text()
    assert "saved" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main(["--check", "-o", str(tmp_path / "no.md")])
    assert not (tmp_path / "no.md").exists()


def test_report_required_sections_and_provenance(recipe):
    script.validate_config(recipe)
    text = (ROOT / "results/p4/REPORT.md").read_text(encoding="utf-8")
    for section in (
        "解析模型要点",
        "校准数据",
        "激活线性性检验",
        "偏差分析",
        "边界声明",
        "复现与验收",
    ):
        assert f"## {section}" in text
    for term in ("CPU", "通信量未实测", "NumPy", "AdamW", "SHA-256", "478720", "tracemalloc"):
        assert term in text

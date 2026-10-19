"""Independent integer arithmetic, bounded round trips, decoder and publishing checks."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prism.kvcache import _forward_cached
from prism.quant import QuantizedSUT, quantize_weight
from prism.sut import DEFAULT_SUT_PATH, SUT, load_sut

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


@pytest.fixture(scope="module")
def bench() -> Any:
    spec = importlib.util.spec_from_file_location("quant_bench", ROOT / "scripts/quant_bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (11, 137)])
def test_int8_round_trip_half_scale_error_bound(shape: tuple[int, int]) -> None:
    source = np.random.default_rng(12).normal(size=shape).astype(np.float32)
    source[-1] = 0
    quant = quantize_weight(source, bits=8)
    actual = quant.dequantize()
    scales = np.max(np.abs(source), axis=1, keepdims=True) / np.float32(127)
    scales[scales == 0] = 1
    np.testing.assert_array_equal(quant.scales, scales)
    assert quant.payload.dtype == np.int8
    assert quant.scales.dtype == actual.dtype == np.float32
    assert quant.scales.shape == (shape[0], 1)
    error = np.abs(actual.astype(np.float64) - source)
    tolerance = quant.scales.astype(np.float64) / 2 + 2 * np.finfo(np.float32).eps * np.abs(source)
    assert np.all(error <= tolerance)
    assert int(quant.payload.min()) >= -127 and int(quant.payload.max()) <= 127
    assert quant.nbytes == source.size + shape[0] * 4
    np.testing.assert_array_equal(actual[-1], 0)


@pytest.mark.parametrize("bits,limit", [(8, 127), (4, 7)])
def test_ties_to_even_and_symmetric_signed_endpoints(bits: Any, limit: int) -> None:
    weight = np.array([[-limit, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, limit]], dtype=np.float32)
    quant = quantize_weight(weight, bits=bits, group_size=8)
    np.testing.assert_array_equal(quant.dequantize(), [[-limit, -2, -2, 0, 0, 2, 2, limit]])
    np.testing.assert_array_equal(quant.scales, 1)
    if bits == 4:
        # [-7,-2,-2,0,0,2,2,7], low nibble first, signed two's complement.
        np.testing.assert_array_equal(quant.payload, np.array([0xE9, 0x0E, 0x20, 0x72], np.uint8))


@pytest.mark.parametrize("group_size", [1, 3, 32, 64, 128])
def test_int4_group_padding_and_independent_unpack(group_size: int) -> None:
    source = np.random.default_rng(1).normal(size=(3, 67)).astype(np.float32)
    source[1] = 0
    quant = quantize_weight(source[:, ::-1], bits=4, group_size=group_size)
    groups = (67 + group_size - 1) // group_size
    assert quant.payload.dtype == np.uint8
    assert quant.payload.size == (3 * groups * group_size + 1) // 2
    assert quant.scales.shape == (3, groups)
    assert quant.nbytes == quant.payload.size + 3 * groups * 4
    # Scalar reference deliberately avoids production vectorized unpack/reshape logic.
    restored = np.zeros((3, 67), dtype=np.float32)
    for row in range(3):
        for column in range(groups * group_size):
            offset = row * groups * group_size + column
            value = (int(quant.payload[offset // 2]) >> (4 * (offset % 2))) & 15
            signed = value - 16 if value >= 8 else value
            assert -7 <= signed <= 7
            if column < 67:
                restored[row, column] = signed * quant.scales[row, column // group_size]
            else:
                assert signed == 0
    actual = quant.dequantize()
    np.testing.assert_array_equal(actual, restored)
    scales = np.repeat(quant.scales, group_size, axis=1)[:, :67]
    error = np.abs(actual.astype(np.float64) - source[:, ::-1])
    assert np.all(error <= scales / 2 + 2 * np.finfo(np.float32).eps * np.abs(source[:, ::-1]))
    assert actual.dtype == np.float32
    assert not quant.payload.flags.writeable and not quant.scales.flags.writeable


@pytest.mark.parametrize("bits", [8, 4])
def test_extreme_finite_and_subnormal_weights(bits: Any) -> None:
    weight = np.array(
        [
            [np.finfo(np.float32).max, -np.finfo(np.float32).max],
            [np.nextafter(np.float32(0), np.float32(1)), 0],
        ],
        dtype=np.float32,
    )
    with np.errstate(all="raise"):
        # nextafter inside quantization itself may signal underflow under this setting.
        with np.errstate(under="ignore"):
            quant = quantize_weight(weight, bits=bits, group_size=2)
            result = quant.dequantize()
    assert np.isfinite(result).all()
    assert np.all(quant.scales > 0)


@pytest.mark.parametrize(
    "weight",
    [
        np.zeros((0, 1), np.float32),
        np.zeros(3, np.float32),
        np.zeros((1, 2), np.float64),
        np.array([[np.nan]], np.float32),
        np.array([[np.inf]], np.float32),
    ],
)
def test_reject_invalid_weights(weight: Any) -> None:
    with pytest.raises(ValueError, match="weight"):
        quantize_weight(weight, bits=8)


@pytest.mark.parametrize("bits,group_size", [(True, 32), (3, 32), (8.0, 32), (4, 0), (4, True)])
def test_reject_invalid_quantization_parameters(bits: Any, group_size: Any) -> None:
    with pytest.raises(ValueError):
        quantize_weight(np.ones((2, 2), np.float32), bits=bits, group_size=group_size)


@pytest.mark.parametrize("bits", [8, 4])
def test_decoder_oracle_memory_and_unchanged_fp32(sut: SUT, bits: Any) -> None:
    tokens = np.array([[256, 316, 32, 396]], dtype=np.int64)
    before = sut.forward(tokens)
    keys = set(sut.arrays)
    originals = {name: array.copy() for name, array in sut.arrays.items()}
    model = QuantizedSUT(sut, bits=bits)
    assert len(model.weights) == sut.config.layers * 4
    assert set(model.arrays).isdisjoint(model.weights)
    for name, array in model.arrays.items():
        assert array is sut.arrays[name]
        assert array.dtype == np.float32
    assert model.arrays["token_embedding.weight"] is sut.arrays["token_embedding.weight"]
    reconstructed = {key: value.dequantize() for key, value in model.weights.items()}
    oracle = SUT(sut.config, model.arrays | reconstructed, sut.tokenizer)
    np.testing.assert_array_equal(model.forward(tokens), oracle.forward(tokens))
    logits, cache = _forward_cached(model, tokens)
    expected, expected_cache = _forward_cached(oracle, tokens)
    np.testing.assert_array_equal(logits, expected)
    following = np.array([[1]], dtype=np.int64)
    logits, cache = _forward_cached(model, following, cache)
    expected, expected_cache = _forward_cached(oracle, following, expected_cache)
    np.testing.assert_array_equal(logits, expected)
    for pair in cache:
        assert all(value.dtype == np.float32 for value in pair)
    ledger = model.memory_ledger()
    assert ledger["total_bytes"] == sum(a.nbytes for a in model.arrays.values()) + sum(
        w.payload.nbytes + w.scales.nbytes for w in model.weights.values()
    )
    assert ledger["fp32_baseline_bytes"] == sum(a.nbytes for a in sut.arrays.values())
    assert ledger["zero_point_bytes"] == 0
    assert ledger["compression_ratio"] > 1
    assert not any(isinstance(value, SUT) for value in vars(model).values())
    assert set(sut.arrays) == keys
    for name, array in sut.arrays.items():
        np.testing.assert_array_equal(array, originals[name])
    np.testing.assert_array_equal(sut.forward(tokens), before)
    # No persistent dequantized weight cache appears after inference.
    assert set(model.arrays).isdisjoint(model.weights)


def test_embedding_quantization_rejected(sut: SUT) -> None:
    with pytest.raises(ValueError, match="quantize_embedding"):
        QuantizedSUT(sut, bits=4, quantize_embedding=True)


def test_recipe_validation_and_suffix_metrics(bench: Any) -> None:
    config = json.loads((ROOT / "configs/p2_quant.json").read_text())
    bench.validate_config(config)
    for change in (
        {"quantize_embedding": True},
        {"group_size": False},
        {"repeat": 0},
        {"threads": 2},
        {"device": "mps"},
    ):
        with pytest.raises(ValueError):
            bench.validate_config(config | change)
    metrics = bench.sequence_metrics([1, 2, 3, 4], [1, 2, 9, 4], 2)
    assert metrics["token_agreement"] == 0.5
    assert metrics["first_divergence_generated_index"] == 0
    assert metrics["first_divergence_absolute_index"] == 2


def test_teacher_forced_argmax_drift_is_diagnostic(
    sut: SUT, bench: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prism.kvcache as kvcache

    original = kvcache._forward_cached

    def near_tie(model: SUT, tokens: Any, past: Any = None) -> Any:
        logits, cache = original(model, tokens, past)
        logits[0, -1] = 0
        logits[0, -1, 1] = 1e-7
        return logits, cache

    monkeypatch.setattr(kvcache, "_forward_cached", near_tie)
    monkeypatch.setattr(
        sut,
        "forward",
        lambda tokens: np.zeros((*tokens.shape, sut.config.vocab_size), dtype=np.float32),
    )
    report = bench.teacher_forced_diagnostics(sut, [256, 1], 1)
    assert report["argmax_disagreements"][0]["prefix_ids"] == [256]
    assert report["argmax_disagreements"][0]["cached_argmax"] == 1


def test_nonfinite_outputs_fail_check(
    sut: SUT, bench: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sut,
        "forward",
        lambda tokens: np.full((*tokens.shape, sut.config.vocab_size), np.nan, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        bench.teacher_forced_diagnostics(sut, [256, 1], 1)


def test_check_executes_recipe_without_writing_and_no_accuracy_gate(
    bench: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prism.record as record

    config = json.loads((ROOT / "configs/p2_quant.json").read_text())
    config.update(
        {
            "sut_path": str(DEFAULT_SUT_PATH),
            "warmup": 0,
            "repeat": 1,
            "decode_tokens": 1,
            "random": {"seed": 42, "count": 1, "prompt_length": 1, "new_tokens": 1},
        }
    )
    for name in (
        "src/prism/quant.py",
        "src/prism/sut.py",
        "src/prism/kvcache.py",
        "scripts/quant_bench.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(record, "save_run", lambda *a, **k: pytest.fail("--check called save_run"))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    summary = bench.run_benchmark(config, check=True)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert summary["correctness"]["implementation_passed"]
    assert summary["correctness"]["accuracy_gate"] is None
    assert len(summary["records"]) == 36
    assert all(len(r["measurement"]["samples_ms"]) == 1 for r in summary["records"])
    assert all(r["path"] is None for r in summary["records"])
    assert all(r["measurement"]["config_hash"] for r in summary["records"])
    assert "Implementation passed does not mean acceptable accuracy" in summary["limitations"]

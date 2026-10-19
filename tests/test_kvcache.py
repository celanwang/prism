"""Exact greedy equivalence, cache boundaries and generation isolation."""

import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prism.kvcache as kvcache
from prism.kvcache import _forward_cached, _generate_cached
from prism.sut import DEFAULT_SUT_PATH, SUT, load_sut


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


@pytest.mark.parametrize(
    ("name", "total", "rebuilds"), [("within_window", 50, 0), ("window_rollover", 143, 14)]
)
def test_cached_matches_independent_fixture(sut: SUT, name: str, total: int, rebuilds: int) -> None:
    fixture = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())
    case = fixture["cases"][name]
    prompt = list(case["prompt_ids"])
    result = _generate_cached(
        sut, prompt, max_new_tokens=case["max_new_tokens"], eos_id=case["eos_id"]
    )
    assert result.token_ids == case["token_ids"]
    assert result.token_ids == sut.generate(prompt, max_new_tokens=case["max_new_tokens"])
    assert result.token_ids == sut.generate(
        prompt, max_new_tokens=case["max_new_tokens"], use_cache=True
    )
    assert len(result.token_ids) == total
    assert result.cache_rebuilds == rebuilds
    assert prompt == case["prompt_ids"]
    positions = min(total - 1, sut.config.context_length)
    assert result.cache_bytes == sut.config.layers * positions * sut.config.width * 2 * 4


@pytest.mark.parametrize("seed", [42, 20261019])
@pytest.mark.parametrize("length", [1, 7, 18, 64, 126, 127, 128, 129, 257])
def test_fixed_random_prompts_match_exactly(sut: SUT, seed: int, length: int) -> None:
    prompt = np.random.default_rng(seed).integers(sut.config.vocab_size, size=length).tolist()
    naive = sut.generate(prompt, max_new_tokens=8, use_cache=False)
    assert sut.generate(prompt, max_new_tokens=8, use_cache=True) == naive


def test_decode_forwards_only_new_tokens_and_rebuilds_full_windows(
    sut: SUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    forward = kvcache._forward_cached

    def tracked_forward(
        model: SUT, tokens: np.ndarray, past: kvcache.KVCache | None = None
    ) -> tuple[np.ndarray, kvcache.KVCache]:
        calls.append((tokens.shape[1], 0 if past is None else past[0][0].shape[2]))
        return forward(model, tokens, past)

    monkeypatch.setattr(kvcache, "_forward_cached", tracked_forward)
    result = _generate_cached(sut, [42] * 127, max_new_tokens=5)
    assert calls == [(127, 0), (1, 127), (128, 0), (128, 0), (128, 0)]
    assert result.cache_rebuilds == 3
    calls.clear()
    _generate_cached(sut, [42] * 129, max_new_tokens=2)
    assert calls == [(128, 0), (128, 0)]


def test_chunked_causal_attention_and_cache_payload(sut: SUT) -> None:
    tokens = np.random.default_rng(7).integers(sut.config.vocab_size, size=(2, 37), dtype=np.int64)
    full = sut.forward(tokens)
    cache = None
    outputs = []
    for start, end in ((0, 3), (3, 4), (4, 19), (19, 37)):
        logits, cache = _forward_cached(sut, tokens[:, start:end], cache)
        outputs.append(logits)
        assert logits.dtype == np.float32
        assert len(cache) == sut.config.layers
        for pair in cache:
            for array in pair:
                assert array.shape == (
                    2,
                    sut.config.heads,
                    end,
                    sut.config.width // sut.config.heads,
                )
                assert array.dtype == np.float32
                assert array.flags.owndata
    actual = np.concatenate(outputs, axis=1)
    assert actual.shape == full.shape
    np.testing.assert_array_equal(actual.argmax(-1), full.argmax(-1))
    # Logit error is a diagnostic, never a numerical pass threshold.
    error = float(np.max(np.abs(actual - full)))
    assert np.isfinite(error)
    assert cache is not None
    assert sum(a.nbytes for pair in cache for a in pair) == 2 * sut.config.layers * 37 * 128 * 2 * 4


def test_cached_eos_zero_length_and_independent_calls(sut: SUT) -> None:
    prompt = [256, 316]
    first = sut.generate(prompt, max_new_tokens=1)[-1]
    assert sut.generate(prompt, max_new_tokens=5, eos_id=first, use_cache=True) == [*prompt, first]
    zero = _generate_cached(sut, prompt, max_new_tokens=0)
    assert zero.token_ids == prompt
    assert zero.token_ids is not prompt
    assert zero.cache_rebuilds == zero.cache_bytes == 0
    assert sut.generate(prompt, max_new_tokens=0, use_cache=True) == prompt
    expected = sut.generate(prompt, max_new_tokens=4)
    for _ in range(2):
        assert sut.generate(prompt, max_new_tokens=4, use_cache=True) == expected
    assert set(vars(sut)) == {"config", "arrays", "tokenizer"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": [], "max_new_tokens": 1},
        {"prompt": [True], "max_new_tokens": 1},
        {"prompt": [512], "max_new_tokens": 1},
        {"prompt": [1], "max_new_tokens": -1},
        {"prompt": [1], "max_new_tokens": True},
        {"prompt": [1], "max_new_tokens": 1, "eos_id": 512},
    ],
)
def test_cached_generation_retains_input_validation(sut: SUT, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sut.generate(**kwargs, use_cache=True)


def test_cached_forward_rejects_overflow_and_incompatible_caches(sut: SUT) -> None:
    token = np.array([[42]], dtype=np.int64)
    _, full = _forward_cached(sut, np.full((1, 128), 42, dtype=np.int64))
    with pytest.raises(ValueError, match="context limit"):
        _forward_cached(sut, token, full)
    _, cache = _forward_cached(sut, token)
    with pytest.raises(ValueError, match="layer count"):
        _forward_cached(sut, token, cache[:1])
    with pytest.raises(ValueError, match="shape or dtype"):
        _forward_cached(sut, np.array([[42], [43]], dtype=np.int64), cache)
    wrong_dtype = tuple((k.astype(np.float64), v) for k, v in cache)
    with pytest.raises(ValueError, match="shape or dtype"):
        _forward_cached(sut, token, wrong_dtype)
    different_lengths = (cache[0], (cache[1][0][:, :, :0], cache[1][1][:, :, :0]))
    with pytest.raises(ValueError, match="positions"):
        _forward_cached(sut, token, different_lengths)


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([1], dtype=np.int64),
        np.array([[1]], dtype=np.int32),
        np.empty((1, 0), dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.array([[512]], dtype=np.int64),
        np.zeros((1, 129), dtype=np.int64),
    ],
)
def test_cached_forward_rejects_invalid_tokens(sut: SUT, tokens: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _forward_cached(sut, tokens)


def test_single_position_context_rebuilds_every_step(sut: SUT) -> None:
    tiny = replace(sut, config=replace(sut.config, context_length=1))
    result = _generate_cached(tiny, [42, 43], max_new_tokens=4)
    assert result.token_ids == tiny.generate([42, 43], max_new_tokens=4)
    assert result.cache_rebuilds == 3


def test_benchmark_saves_both_orders_and_validated_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism.record import load_run

    root = Path(__file__).resolve().parents[1]
    bench = runpy.run_path(str(root / "scripts" / "kvcache_bench.py"))
    config = json.loads((root / "configs" / "p1_kvcache.json").read_text())
    config["decode"].update(warmup=0, repeat=1)
    # This is a persistence/protocol smoke test, not performance evidence.
    for variable in bench["THREAD_VARIABLES"]:
        monkeypatch.setenv(variable, "1")
    summary = bench["run_benchmark"](config, tmp_path)
    runs = sorted(tmp_path.iterdir())
    assert len(runs) == 4
    assert [row["order"] for row in summary["comparisons"]] == config["orders"]
    assert summary["correctness"]["cases"]["window_rollover"]["cache_rebuilds"] == 14
    assert summary["correctness"]["full_window_array_bytes"] == 262144
    for row in summary["comparisons"]:
        for arm, record in row["arms"].items():
            run = load_run(record["path"])
            assert run.config["order"] == row["order"]
            assert run.config["workload"]["use_cache"] == (arm == "cached")
            assert len(run.measurement.samples_ms) == 1
            assert run.measurement.config_hash == record["config_hash"]
            assert run.measurement.p50 == record["p50_ms"]
    saved = json.loads(
        (Path(summary["comparisons"][0]["arms"]["naive"]["path"]) / "comparison.json").read_text()
    )
    assert saved == summary


def test_benchmark_rejects_missing_counterbalance_before_writing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bench = runpy.run_path(str(root / "scripts" / "kvcache_bench.py"))
    config = json.loads((root / "configs" / "p1_kvcache.json").read_text())
    config["orders"] = [["naive", "cached"]]
    with pytest.raises(ValueError, match="orders"):
        bench["run_benchmark"](config, tmp_path)
    assert not list(tmp_path.iterdir())

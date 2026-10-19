"""Shared token oracle, physical page lifetime, eviction misses and run provenance."""

from __future__ import annotations

import json
import math
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prism.paged import InfeasibleBudget, PagedKVCache, forward_paged, generate_paged
from prism.record import load_run
from prism.sut import DEFAULT_SUT_PATH, SUT, load_sut

ROOT = Path(__file__).resolve().parents[1]
BENCH = runpy.run_path(str(ROOT / "scripts" / "paged_bench.py"))


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


@pytest.mark.parametrize(
    "arm,block", [("naive", 8), ("continuous", 8), ("paged", 8), ("paged", 16), ("paged", 32)]
)
@pytest.mark.parametrize(
    "name,total,rebuilds", [("within_window", 50, 0), ("window_rollover", 143, 14)]
)
def test_three_adapters_use_same_per_token_oracle(
    sut: SUT, arm: str, block: int, name: str, total: int, rebuilds: int
) -> None:
    fixture = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())
    result = BENCH["check_case"](sut, fixture["cases"][name], arm, block_size=block)
    assert result["matched_tokens"] == total
    assert result["cache_rebuilds"] == (0 if arm == "naive" else rebuilds)
    assert np.isfinite(result["max_abs_logit_error_vs_naive"])


@pytest.mark.parametrize("block", [8, 16, 32])
@pytest.mark.parametrize("seed", [42, 20261019])
@pytest.mark.parametrize("length", [1, 7, 18, 64, 126, 127, 128, 129, 257])
def test_random_prompts_match_naive_and_continuous(
    sut: SUT, block: int, seed: int, length: int
) -> None:
    prompt = np.random.default_rng(seed).integers(sut.config.vocab_size, size=length).tolist()
    expected = sut.generate(prompt, max_new_tokens=8)
    assert sut.generate(prompt, max_new_tokens=8, use_cache=True) == expected
    actual = generate_paged(sut, prompt, max_new_tokens=8, block_size=block, budget_bytes=262144)
    assert actual.token_ids == expected
    assert actual.cache_rebuilds == max(0, min(7, length + 6 - 127))
    assert len(prompt) == length


@pytest.mark.parametrize("block", [1, 8, 16, 32, 31, 256])
def test_chunked_batch_forward_gathers_exact_pages(sut: SUT, block: int) -> None:
    tokens = np.random.default_rng(7).integers(sut.config.vocab_size, size=(2, 37), dtype=np.int64)
    cache = PagedKVCache(sut.config, block_size=block, budget_bytes=1048576, batch_size=2)
    outputs = []
    for start, end in ((0, 3), (3, 4), (4, 19), (19, 37)):
        outputs.append(forward_paged(sut, tokens[:, start:end], cache))
        assert cache.length == end
        stats = cache.stats
        assert stats.allocated_bytes == math.ceil(end / block) * block * 4096
        assert stats.used_bytes == end * 4096
        assert len(set(cache.page_table.values())) == stats.resident_pages
        assert stats.allocated_bytes <= cache.budget_bytes
    actual = np.concatenate(outputs, axis=1)
    np.testing.assert_array_equal(actual.argmax(-1), sut.forward(tokens).argmax(-1))
    assert np.isfinite(np.max(np.abs(actual - sut.forward(tokens))))
    # Gathers are detached copies, unaffected by later physical-slot recycling.
    before = cache.gather(0, 37)
    saved = tuple(a.copy() for a in before)
    table = cache.page_table
    table.clear()
    assert cache.page_table
    cache.clear()
    assert cache.stats.allocated_bytes == 0
    assert cache.stats.releases == math.ceil(37 / block)
    forward_paged(sut, np.zeros((2, 37), dtype=np.int64), cache)
    for old, expected in zip(before, saved, strict=True):
        np.testing.assert_array_equal(old, expected)


def test_rollover_invalidates_all_pages_before_full_rebuild(
    sut: SUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prism.paged as paged

    calls = []
    released = []
    forward = paged.forward_paged
    release = PagedKVCache.release

    def tracked(model: SUT, tokens: np.ndarray, cache: PagedKVCache) -> np.ndarray:
        calls.append((tokens.shape[1], cache.length, cache.stats.resident_pages))
        return forward(model, tokens, cache)

    def track_release(self: PagedKVCache, logical: int, *, eviction: bool = False) -> None:
        released.append(logical)
        release(self, logical, eviction=eviction)

    monkeypatch.setattr(paged, "forward_paged", tracked)
    monkeypatch.setattr(PagedKVCache, "release", track_release)
    result = generate_paged(sut, [42] * 127, max_new_tokens=5, block_size=8, budget_bytes=262144)
    assert calls == [(127, 0, 0), (1, 127, 16), (128, 0, 0), (128, 0, 0), (128, 0, 0)]
    assert result.cache_rebuilds == 3
    assert released == list(range(16)) * 3
    assert result.stats.evictions == result.stats.releases == 48
    assert result.stats.allocations == 64
    assert result.stats.peak_allocated_bytes == 262144
    assert result.token_ids == sut.generate([42] * 127, max_new_tokens=5)


@pytest.mark.parametrize("block,budget", [(8, 0), (8, 16383), (16, 114688), (32, 114688)])
def test_infeasible_budget_fails_before_any_model_forward(
    sut: SUT, monkeypatch: pytest.MonkeyPatch, block: int, budget: int
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("infeasible request must not forward or fall back")

    monkeypatch.setattr(SUT, "forward", unexpected)
    monkeypatch.setattr("prism.paged.forward_paged", unexpected)
    with pytest.raises(InfeasibleBudget, match="complete correct window"):
        generate_paged(sut, [42] * 18, max_new_tokens=32, block_size=block, budget_bytes=budget)


def test_forward_capacity_failure_is_atomic(sut: SUT) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=16384)
    forward_paged(sut, np.full((1, 8), 42, dtype=np.int64), cache)
    table = cache.page_table
    key, value = cache.gather(0, 8)
    # Gather lookup counters intentionally differ from the previous snapshot.
    before = asdict(cache.stats)
    with pytest.raises(InfeasibleBudget):
        forward_paged(sut, np.array([[42]], dtype=np.int64), cache)
    assert cache.page_table == table
    assert asdict(cache.stats) == before
    actual = cache.gather(0, 8)
    np.testing.assert_array_equal(actual[0], key)
    np.testing.assert_array_equal(actual[1], value)
    assert cache.length == 8


@pytest.mark.parametrize("policy,evicted", [("sliding", 0), ("lru", 1)])
def test_eviction_miss_recomputes_and_recycles_without_stale_data(
    sut: SUT, policy: str, evicted: int
) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=32768, policy=policy)
    recomputed = []

    def payload(logical: int) -> np.ndarray:
        recomputed.append(logical)
        return np.full((2, 2, 1, 4, 8, 32), logical + 0.25, dtype=np.float32)

    assert not cache.access_block(0, 8, payload)
    slot = cache.page_table[0]
    assert not cache.access_block(1, 8, payload)
    assert cache.access_block(0, 8, payload)
    assert not cache.access_block(2, 8, payload)
    assert evicted not in cache.page_table
    assert cache.stats.evictions == 1
    if policy == "sliding":
        assert cache.page_table[2] == slot
    assert not cache.access_block(evicted, 8, payload)
    assert recomputed == [0, 1, 2, evicted]
    assert cache.stats.hits == 1 and cache.stats.misses == 4
    assert cache.stats.evictions == 2
    assert cache.stats.peak_allocated_bytes == cache.stats.allocated_bytes == 32768
    if policy == "sliding":
        key, value = cache.gather(0, 8)
        np.testing.assert_array_equal(key, np.full_like(key, 0.25))
        np.testing.assert_array_equal(value, key)
    else:
        # Restore a contiguous synthetic prefix to inspect the recomputed miss.
        cache.access_block(0, 8, payload)
        key, value = cache.gather(1, 16)
        np.testing.assert_array_equal(key[:, :, :8], np.full_like(key[:, :, :8], 0.25))
        np.testing.assert_array_equal(key[:, :, 8:], np.full_like(key[:, :, 8:], 1.25))
        np.testing.assert_array_equal(value, key)
        with pytest.raises(ValueError, match="simulation only"):
            forward_paged(sut, np.array([[42]], dtype=np.int64), cache)


def test_unwritten_missing_and_partial_pages_never_gather_padding(sut: SUT) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=32768)
    cache.reserve(7)
    with pytest.raises(ValueError, match="initialized"):
        cache.gather(0, 7)
    cache.clear()
    forward_paged(sut, np.full((1, 7), 42, dtype=np.int64), cache)
    with pytest.raises(ValueError, match="initialized"):
        cache.gather(0, 8)
    forward_paged(sut, np.array([[42]], dtype=np.int64), cache)
    with pytest.raises(KeyError, match="missing"):
        cache.gather(0, 9)
    cache.release(0)
    with pytest.raises(KeyError):
        cache.release(0)
    assert cache.stats.allocated_bytes == cache.stats.used_bytes == 0
    assert cache.stats.hit_rate is not None


def test_failed_recompute_does_not_evict_resident_pages(sut: SUT) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=16384, policy="lru")
    payload = np.zeros((2, 2, 1, 4, 8, 32), dtype=np.float32)
    cache.access_block(0, 8, lambda _: payload)
    with pytest.raises(ValueError, match="payload"):
        cache.access_block(1, 8, lambda _: payload.astype(np.float64))
    assert cache.page_table == {0: 0}
    assert cache.stats.evictions == 0


@pytest.mark.parametrize("length", [1, 7, 8, 18, 49, 127, 128])
def test_fixed_length_nested_blocks_have_monotone_internal_fragmentation(
    sut: SUT, length: int
) -> None:
    fragments = []
    for block in (8, 16, 32):
        cache = PagedKVCache(sut.config, block_size=block, budget_bytes=262144)
        forward_paged(sut, np.full((1, length), 42, dtype=np.int64), cache)
        stats = cache.stats
        assert stats.allocated_bytes == math.ceil(length / block) * block * cache.bytes_per_token
        assert stats.used_bytes == length * cache.bytes_per_token
        fragments.append(stats.fragmentation)
    assert fragments == sorted(fragments)


def test_eos_zero_generation_single_position_context_and_isolation(sut: SUT) -> None:
    prompt = [256, 316]
    first = sut.generate(prompt, max_new_tokens=1)[-1]
    result = generate_paged(
        sut, prompt, max_new_tokens=5, eos_id=first, block_size=8, budget_bytes=16384
    )
    assert result.token_ids == [*prompt, first]
    zero = generate_paged(sut, prompt, max_new_tokens=0, block_size=8, budget_bytes=0)
    assert zero.token_ids == prompt and zero.token_ids is not prompt
    assert zero.cache_rebuilds == zero.stats.allocated_bytes == 0
    assert zero.stats.hit_rate is None
    tiny = replace(sut, config=replace(sut.config, context_length=1))
    for _ in range(2):
        result = generate_paged(tiny, [42, 43], max_new_tokens=4, block_size=8, budget_bytes=16384)
        assert result.token_ids == tiny.generate([42, 43], max_new_tokens=4)
        assert result.cache_rebuilds == 3
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
def test_generation_validates_like_sut(sut: SUT, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        generate_paged(sut, **kwargs, block_size=8, budget_bytes=262144)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0},
        {"block_size": True},
        {"budget_bytes": -1},
        {"budget_bytes": True},
        {"batch_size": 0},
        {"policy": "unknown"},
    ],
)
def test_invalid_pool_settings(sut: SUT, kwargs: dict[str, Any]) -> None:
    settings = {"block_size": 8, "budget_bytes": 262144, **kwargs}
    with pytest.raises(ValueError):
        PagedKVCache(sut.config, **settings)


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([1], dtype=np.int64),
        np.array([[1]], dtype=np.int32),
        np.empty((1, 0), dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.array([[512]], dtype=np.int64),
        np.zeros((1, 129), dtype=np.int64),
        np.ones((2, 1), dtype=np.int64),
    ],
)
def test_forward_rejects_invalid_tokens(sut: SUT, tokens: np.ndarray) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=262144)
    with pytest.raises(ValueError):
        forward_paged(sut, tokens, cache)
    assert cache.stats.allocations == 0


def test_forward_exception_invalidates_partial_layer_updates(
    sut: SUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = PagedKVCache(sut.config, block_size=8, budget_bytes=262144)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(SUT, "_linear", fail)
    with pytest.raises(RuntimeError, match="injected"):
        forward_paged(sut, np.array([[42]], dtype=np.int64), cache)
    assert cache.length == cache.stats.allocated_bytes == 0
    assert cache.page_table == {}


def test_benchmark_persists_infeasibility_simulations_and_both_generation_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads((ROOT / "configs" / "p1_paged.json").read_text())
    config["block_sizes"] = [8, 16]
    config["budgets_bytes"] = [65536, 114688, 131072]
    for key in ("decode", "allocation", "simulation"):
        config[key].update(warmup=0, repeat=1)
    for name in BENCH["THREAD_VARIABLES"]:
        monkeypatch.setenv(name, "1")
    # Persistence smoke only; performance evidence uses the unchanged checked-in recipe.
    summary = BENCH["run_benchmark"](config, tmp_path)
    assert summary["correctness"]["cases"]["window_rollover"]["paged-b8"]["cache_rebuilds"] == 14
    assert [c["order"] for c in summary["comparisons"]] == config["orders"]
    assert len(summary["points"]) == 6
    feasible = [p for p in summary["points"] if p["status"] == "feasible"]
    assert len(feasible) == 3
    for point in summary["points"]:
        assert point["actual_minus_analytic_bytes"] == 0
        assert point["allocation_stats"]["peak_allocated_bytes"] <= point["budget_bytes"]
        if point["status"] == "infeasible":
            assert "generation_stats" not in point
            assert "throughput" not in point
        assert point["simulations"]["lru"]["evidence"] == "memory simulation only"
    paths = []
    for comparison in summary["comparisons"]:
        assert len(comparison["arms"]["paged"]) == 3
        for record in [
            comparison["arms"]["naive"],
            comparison["arms"]["continuous"],
            *comparison["arms"]["paged"],
        ]:
            paths.append(record["path"])
            run = load_run(record["path"])
            assert run.config["workload"]["order"] == comparison["order"]
            assert run.measurement.config_hash == record["config_hash"]
            assert run.measurement.env_fingerprint == record["env_fingerprint"]
            assert len(run.measurement.samples_ms) == 1
    for point in summary["points"]:
        paths.append(point["allocation_measurement"]["path"])
        for sim in point["simulations"].values():
            paths.append(sim["measurement"]["path"])
    assert len(paths) == len(list(tmp_path.iterdir())) == 28
    for path in paths:
        run = load_run(path)
        assert run.path.name.startswith(datetime_prefix() + "-p1-paged-")
        assert len(run.measurement.samples_ms) == 1
    first = summary["points"][0]["allocation_measurement"]["path"]
    saved = json.loads((Path(first) / "summary.json").read_text())
    assert saved == summary


def datetime_prefix() -> str:
    """Use UTC for the run prefix even when the workstation date differs."""
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d")


def test_benchmark_invalid_recipe_writes_nothing(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "p1_paged.json").read_text())
    config["orders"] = [["naive", "continuous", "paged"]]
    with pytest.raises(ValueError, match="orders"):
        BENCH["run_benchmark"](config, tmp_path)
    assert list(tmp_path.iterdir()) == []

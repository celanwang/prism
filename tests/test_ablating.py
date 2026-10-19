"""Fixture token gates, isolated recipes, buffer ownership and T4 decisions."""

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prism.ablating import VARIANTS, AblatingSUT
from prism.kvcache import _forward_cached
from prism.measure import measure
from prism.sut import DEFAULT_SUT_PATH, SUT, array_digest, load_sut

_SPEC = importlib.util.spec_from_file_location(
    "kernel_bench", Path(__file__).resolve().parents[1] / "scripts/kernel_bench.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case_name", ["within_window", "window_rollover"])
@pytest.mark.parametrize("cached", [False, True])
def test_every_fixture_token(sut: SUT, variant: str, case_name: str, cached: bool) -> None:
    case = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())["cases"][case_name]
    model = AblatingSUT(sut, variant)
    actual = model.generate(
        case["prompt_ids"],
        max_new_tokens=case["max_new_tokens"],
        eos_id=case["eos_id"],
        use_cache=cached,
    )
    assert actual == case["token_ids"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_batched_logits_cache_and_causality(sut: SUT, variant: str) -> None:
    tokens = np.random.default_rng(24).integers(0, 512, size=(2, 9), dtype=np.int64)
    model = AblatingSUT(sut, variant)
    original = sut.forward(tokens)
    actual = model.forward(tokens)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, original, atol=3e-5, rtol=3e-5)
    if variant == "baseline":
        np.testing.assert_array_equal(actual, original)
    logits, cache = model.forward_cached(tokens[:, :5])
    expected, expected_cache = _forward_cached(sut, tokens[:, :5])
    np.testing.assert_allclose(logits, expected, atol=3e-5, rtol=3e-5)
    for end in range(5, 9):
        logits, cache = model.forward_cached(tokens[:, end : end + 1], cache)
        expected, expected_cache = _forward_cached(sut, tokens[:, end : end + 1], expected_cache)
        np.testing.assert_allclose(logits, expected, atol=3e-5, rtol=3e-5)
        if variant == "baseline":
            np.testing.assert_array_equal(logits, expected)
        for pair, reference in zip(cache, expected_cache, strict=True):
            for a, b in zip(pair, reference, strict=True):
                np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
    changed = tokens.copy()
    changed[:, -1] = 3
    np.testing.assert_array_equal(actual[:, :-1], model.forward(changed)[:, :-1])


def test_scratch_reuse_ownership_and_shape_changes(sut: SUT) -> None:
    model = AblatingSUT(sut, "mlp_reuse")
    before = array_digest(*sut.arrays.values())
    tokens = np.array([[256, 316, 32]], dtype=np.int64)
    first, cache = model.forward_cached(tokens)
    first_copy = first.copy()
    cache_copies = tuple(tuple(a.copy() for a in pair) for pair in cache)
    scratch = model._scratch
    assert scratch is not None
    model.forward(tokens + 1)
    assert all(a is b for a, b in zip(scratch, model._scratch, strict=True))
    np.testing.assert_array_equal(first, first_copy)
    for pair, expected in zip(cache, cache_copies, strict=True):
        for a, b in zip(pair, expected, strict=True):
            np.testing.assert_array_equal(a, b)
            assert all(not np.shares_memory(a, s) for s in scratch)
    model.forward(tokens[:, :1])
    assert model._scratch[0].shape == (1, 1, 4 * sut.config.width)
    model.forward(tokens)
    np.testing.assert_array_equal(model.forward(tokens), first)
    assert before == array_digest(*sut.arrays.values())


def test_only_selected_switch_runs(sut: SUT, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = SUT._linear

    def linear(self: SUT, x: Any, name: str) -> Any:
        calls.append(name)
        return original(self, x, name)

    monkeypatch.setattr(SUT, "_linear", linear)
    for variant in VARIANTS:
        calls.clear()
        model = AblatingSUT(sut, variant)
        model.forward(np.array([[256, 316]], dtype=np.int64))
        assert sum(n.endswith("attention.qkv") for n in calls) == (
            0 if variant == "qkv_split" else sut.config.layers
        )
        assert (model._scratch is not None) == (variant == "mlp_reuse")
        assert sum(n.endswith(".up") for n in calls) == (
            0 if variant == "mlp_reuse" else sut.config.layers
        )
    with pytest.raises(ValueError, match="exactly one"):
        AblatingSUT(sut, "qkv_split+mlp_reuse")


@pytest.mark.parametrize("variant", VARIANTS)
def test_input_eos_and_cache_validation(sut: SUT, variant: str) -> None:
    model = AblatingSUT(sut, variant)
    for tokens in (
        np.array([[1]], dtype=np.int32),
        np.empty((1, 0), dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.zeros((1, 129), dtype=np.int64),
    ):
        with pytest.raises(ValueError):
            model.forward(tokens)
        with pytest.raises(ValueError):
            model.forward_cached(tokens)
    prompt = [256, 316]
    following = model.generate(prompt, max_new_tokens=1)[-1]
    assert model.generate(prompt, max_new_tokens=4, eos_id=following, use_cache=True) == [
        *prompt,
        following,
    ]
    assert model.generate(prompt, max_new_tokens=0, use_cache=True) == prompt
    with pytest.raises(ValueError):
        model.generate([], max_new_tokens=1, use_cache=True)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=-1, use_cache=True)
    with pytest.raises(ValueError, match="layer count"):
        model.forward_cached(np.array([[1]], dtype=np.int64), ())
    _, cache = model.forward_cached(np.zeros((1, 128), dtype=np.int64))
    with pytest.raises(ValueError, match="context"):
        model.forward_cached(np.array([[1]], dtype=np.int64), cache)


def test_recipe_loading_and_exact_t4_parameters() -> None:
    config = bench.load_config()
    assert {r["variant"] for r in config["recipes"]} == set(VARIANTS) - {"baseline"}
    for recipe in config["recipes"]:
        assert (recipe["warmup"], recipe["repeat"]) == (100, 2000)
        assert recipe["prefill"]["length"] == 128
        assert recipe["decode"]["tokens"] == 32


@pytest.mark.parametrize(
    "change", ["bundle", "duplicate", "orders", "prefill", "decode", "repeat", "t4"]
)
def test_invalid_recipes_fail_closed(change: str) -> None:
    config = copy.deepcopy(bench.load_config())
    recipe = config["recipes"][0]
    if change == "bundle":
        recipe["variant"] = ["qkv_split", "mlp_reuse"]
    elif change == "duplicate":
        config["recipes"][1] = copy.deepcopy(recipe)
    elif change == "orders":
        recipe["orders"][0].append("attention_bt")
    elif change == "prefill":
        recipe["prefill"]["length"] = 127
    elif change == "decode":
        recipe["decode"]["tokens"] = 31
    elif change == "repeat":
        recipe["repeat"] = True
    else:
        config["calibration"]["resolution_ms"] = 0
    with pytest.raises(ValueError):
        bench.validate_config(config)


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ([0.092583, 1.0], "不可分辨"),
        ([0.092584, 0.2], "值得"),
        ([-0.1, -0.092584], "不值得"),
        ([0.2, -0.2], "不可分辨"),
        ([0.0, 0.0], "不可分辨"),
        ([-0.092583, -0.3], "不可分辨"),
    ],
)
def test_t4_threshold_both_orders(deltas: list[float], expected: str) -> None:
    assert bench.threshold_verdict(deltas, 0.092584) == expected


@pytest.mark.parametrize("deltas", [[float("nan"), 1], [float("inf"), 1], [1]])
def test_t4_rejects_invalid_deltas(deltas: list[float]) -> None:
    with pytest.raises(ValueError):
        bench.threshold_verdict(deltas, 0.092584)


def test_benchmark_calls_p0_protocol_and_saves_isolated_runs(
    sut: SUT, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from prism import measure as measurement_module
    from prism import record

    calls = []
    saves = []
    dummy = measure(lambda: None, warmup=0, repeat=1, config={})

    def measured(fn: Any, *, warmup: int, repeat: int, config: dict[str, Any]) -> Any:
        logits, cache = fn()
        assert np.isfinite(logits).all()
        assert len(cache) == sut.config.layers
        calls.append((warmup, repeat, config))
        return dummy

    def save(runs_dir: Path, slug: str, config: dict[str, Any], m: Any) -> Path:
        saves.append((slug, config))
        path = tmp_path / f"20261019-{slug}"
        path.mkdir()
        return path

    # Keep real inference and fixture gates; only replace expensive measurement/storage.
    monkeypatch.setattr(measurement_module, "measure", measured)
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    config = bench.load_config()
    config["sut_path"] = str(DEFAULT_SUT_PATH)
    for name in (
        "src/prism/ablating.py",
        "src/prism/sut.py",
        "src/prism/kvcache.py",
        "src/prism/measure.py",
        "scripts/kernel_bench.py",
    ):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("source hash fixture")
    # load_run is not exercised here: its existing tests cover persistence integrity.
    from types import SimpleNamespace

    monkeypatch.setattr(record, "save_run", save)
    monkeypatch.setattr(record, "load_run", lambda path: SimpleNamespace(measurement=dummy))
    summary = bench.run_benchmark(config, recipe_name="qkv")
    assert len(calls) == len(saves) == len(summary["records"]) == 8
    assert all((w, r) == (100, 2000) for w, r, _ in calls)
    assert {c["workload"]["name"] for _, _, c in calls} == {"prefill", "decode"}
    assert {c["workload"]["arm"] for _, _, c in calls} == {"baseline", "qkv_split"}
    assert all(c["recipe"]["name"] == "qkv" and "recipes" not in c for _, _, c in calls)
    assert summary["correctness"]["passed"]
    assert all(c["verdict"] == "不可分辨" for c in summary["conclusions"])
    assert (tmp_path / summary["records"][0]["path"] / "comparison.json").is_file()
    saves.clear()
    checked = bench.run_benchmark(config, recipe_name="qkv", check=True)
    assert not saves
    assert all(r["path"] is None for r in checked["records"])

"""Hard greedy matrix plus independent rejection, EOS, boundary and record checks."""

import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prism.speculative as speculative
from prism.kvcache import KVCache
from prism.quant import QuantizedSUT
from prism.speculative import SpeculativeResult, generate_speculative
from prism.sut import DEFAULT_SUT_PATH, SUT, FloatArray, load_sut

ROOT = Path(__file__).resolve().parents[1]
KS = [1, 2, 4, 8]
LENGTHS = [1, 7, 18, 64, 126, 127, 128, 129, 257]


@pytest.fixture(scope="module")
def models() -> tuple[SUT, QuantizedSUT]:
    target = load_sut(DEFAULT_SUT_PATH)
    return target, QuantizedSUT(target, bits=8)


def audit(result: SpeculativeResult, prompt: list[int], k: int) -> None:
    """Reconstruct outputs and counts from independently examined per-position logs."""
    tokens = list(prompt)
    histogram = {str(i): 0 for i in range(k + 1)}
    proposed = examined = accepted = fallback = advance = rounds = 0
    for step in result.steps:
        assert step.start == len(tokens)
        assert step.emitted
        if step.mode == "target_only":
            assert not step.candidates and not step.targets and step.accepted == 0
            assert len(step.emitted) == 1 and not step.bonus
            fallback += 1
        else:
            rounds += 1
            count = 0
            for candidate, target in zip(step.candidates, step.targets, strict=False):
                if candidate != target:
                    break
                count += 1
            assert step.accepted == count
            assert step.emitted[: len(step.targets)] == step.targets
            assert len(step.emitted) == len(step.targets) + int(step.bonus)
            assert not step.bonus or step.accepted == len(step.candidates)
            histogram[str(count)] += 1
            proposed += len(step.candidates)
            examined += len(step.targets)
            accepted += count
            advance += len(step.emitted)
        tokens.extend(step.emitted)
    assert result.token_ids == tokens
    stats = result.statistics(k)
    assert stats["accepted_per_round_histogram"] == histogram
    assert (stats["proposed"], stats["examined"], stats["accepted"], stats["fallback_tokens"]) == (
        proposed,
        examined,
        accepted,
        fallback,
    )
    assert stats["mean_speculative_advance"] == (advance / rounds if rounds else None)
    for i, position in enumerate(stats["positions"]):
        spec = [s for s in result.steps if s.mode == "speculative"]
        assert position["proposed"] == sum(len(s.candidates) > i for s in spec)
        assert position["examined"] == sum(len(s.targets) > i for s in spec)
        assert position["accepted"] == sum(s.accepted > i for s in spec)


@pytest.mark.parametrize("name", ["within_window", "window_rollover"])
@pytest.mark.parametrize("k", KS)
def test_fixture_hard_gate(models: tuple[SUT, QuantizedSUT], name: str, k: int) -> None:
    target, draft = models
    case = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())["cases"][name]
    prompt = list(case["prompt_ids"])
    result = generate_speculative(target, draft, prompt, max_new_tokens=case["max_new_tokens"], k=k)
    assert result.token_ids == case["token_ids"]
    assert result.token_ids == target.generate(
        prompt, max_new_tokens=case["max_new_tokens"], use_cache=True
    )
    assert prompt == case["prompt_ids"]
    audit(result, prompt, k)
    if name == "window_rollover":
        assert result.statistics(k)["fallback_tokens"] > 0


@pytest.mark.parametrize("seed", [42, 20261019])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("k", KS)
def test_fixed_random_full_matrix(
    models: tuple[SUT, QuantizedSUT], seed: int, length: int, k: int
) -> None:
    target, draft = models
    prompt = np.random.default_rng(seed).integers(target.config.vocab_size, size=length).tolist()
    baseline = target.generate(prompt, max_new_tokens=8, use_cache=True)
    assert baseline == target.generate(prompt, max_new_tokens=8)
    result = generate_speculative(target, draft, prompt, max_new_tokens=8, k=k)
    assert result.token_ids == baseline
    audit(result, prompt, k)


def fake_forward(
    monkeypatch: pytest.MonkeyPatch, target: SUT, rejects: set[int]
) -> list[tuple[bool, int, list[int]]]:
    """Synthetic deterministic decoder only for control-flow tests; no real weight changes.

    Prediction depends on the entire cached prefix, so stale rejected K/V changes
    later output. Cache payload encodes input tokens, independent of implementation.
    """
    calls = []

    def forward(
        model: SUT, tokens: np.ndarray, past: KVCache | None = None
    ) -> tuple[FloatArray, KVCache]:
        prefix = [] if past is None else past[0][0][0, 0, :, 0].astype(int).tolist()
        calls.append((model is target, len(prefix), tokens[0].tolist()))
        outputs = []
        for token in tokens[0]:
            prefix.append(int(token))
            following = (sum(prefix) + len(prefix)) % model.config.vocab_size
            if model is not target and len(prefix) in rejects:
                following = (following + 1) % model.config.vocab_size
            logits = np.full(model.config.vocab_size, -1, dtype=np.float32)
            logits[following] = 1
            outputs.append(logits)
        key = np.array(prefix, dtype=np.float32).reshape(1, 1, -1, 1)
        return np.array([outputs], dtype=np.float32), ((key, key.copy()),)

    monkeypatch.setattr(speculative, "_forward_cached", forward)
    return calls


def reference(prompt: list[int], count: int, vocabulary: int, eos: int | None = None) -> list[int]:
    """Independent scalar greedy oracle for the synthetic decoder."""
    tokens = list(prompt)
    for _ in range(count):
        following = (sum(tokens) + len(tokens)) % vocabulary
        tokens.append(following)
        if following == eos:
            break
    return tokens


@pytest.mark.parametrize("rejected_position", range(4))
def test_rejection_rolls_back_both_caches(
    models: tuple[SUT, QuantizedSUT], monkeypatch: pytest.MonkeyPatch, rejected_position: int
) -> None:
    target, draft = models
    prompt = [5, 7]
    calls = fake_forward(monkeypatch, target, {len(prompt) + rejected_position})
    result = generate_speculative(target, draft, prompt, max_new_tokens=18, k=4)
    assert result.token_ids == reference(prompt, 18, target.config.vocab_size)
    assert result.steps[0].accepted == rejected_position
    assert not result.steps[0].bonus
    # Target makes exactly one verification call per speculative round (plus prefill).
    assert sum(is_target for is_target, _, _ in calls) == len(result.steps) + 1
    for is_target, _, tokens in calls[1:]:
        if is_target:
            assert 2 <= len(tokens) <= 5
    audit(result, prompt, 4)


@pytest.mark.parametrize("budget", [0, 1, 3, 4, 5, 6, 9, 10])
def test_budget_bonus_and_terminal_round(
    models: tuple[SUT, QuantizedSUT], monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    target, draft = models
    calls = fake_forward(monkeypatch, target, set())
    result = generate_speculative(target, draft, [5], max_new_tokens=budget, k=4)
    assert result.token_ids == reference([5], budget, target.config.vocab_size)
    assert sum(len(s.emitted) for s in result.steps) == budget
    if budget == 0:
        assert not calls
    elif budget <= 4:
        assert len(result.steps) == 1 and not result.steps[0].bonus
        assert len(result.steps[0].candidates) == budget
    else:
        assert result.steps[0].bonus
    audit(result, [5], 4)


@pytest.mark.parametrize("eos_position", range(6))
def test_eos_in_accepted_draft_or_bonus(
    models: tuple[SUT, QuantizedSUT], monkeypatch: pytest.MonkeyPatch, eos_position: int
) -> None:
    target, draft = models
    fake_forward(monkeypatch, target, set())
    expected = reference([5], 10, target.config.vocab_size)
    eos = expected[1 + eos_position]
    result = generate_speculative(target, draft, [5], max_new_tokens=10, k=4, eos_id=eos)
    assert result.token_ids == reference([5], 10, target.config.vocab_size, eos)
    audit(result, [5], 4)


def test_replacement_eos(models: tuple[SUT, QuantizedSUT], monkeypatch: pytest.MonkeyPatch) -> None:
    target, draft = models
    fake_forward(monkeypatch, target, {1})
    result = generate_speculative(target, draft, [5], max_new_tokens=8, k=4, eos_id=6)
    assert result.token_ids == [5, 6]
    assert result.steps[0].accepted == 0 and not result.steps[0].bonus
    audit(result, [5], 4)


def test_window_calls_and_immutability(
    models: tuple[SUT, QuantizedSUT], monkeypatch: pytest.MonkeyPatch
) -> None:
    target, draft = models
    before = {name: value.tobytes() for name, value in target.arrays.items()}
    prompt = [42] * 127
    forward = speculative._forward_cached
    calls = []

    def tracked(
        model: SUT, tokens: np.ndarray, past: KVCache | None = None
    ) -> tuple[FloatArray, KVCache]:
        calls.append((model is target, tokens.shape[1], 0 if past is None else past[0][0].shape[2]))
        return forward(model, tokens, past)

    monkeypatch.setattr(speculative, "_forward_cached", tracked)
    result = generate_speculative(target, draft, prompt, max_new_tokens=5, k=8)
    assert result.token_ids == target.generate(prompt, max_new_tokens=5, use_cache=True)
    assert calls[:4] == [(True, 126, 0), (False, 126, 0), (False, 1, 126), (True, 2, 126)]
    assert calls[4:] == [(True, 128, 0)] * 3
    assert before == {name: value.tobytes() for name, value in target.arrays.items()}
    assert prompt == [42] * 127
    assert generate_speculative(target, draft, prompt, max_new_tokens=5, k=8) == result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": True},
        {"max_new_tokens": -1},
        {"max_new_tokens": True},
        {"prompt": []},
        {"prompt": [True]},
        {"prompt": [-1]},
        {"prompt": [99999]},
        {"eos_id": True},
        {"eos_id": -1},
    ],
)
def test_invalid_generation_inputs(
    models: tuple[SUT, QuantizedSUT], kwargs: dict[str, Any]
) -> None:
    target, draft = models
    args = {"prompt": [5], "max_new_tokens": 1, "k": 4, **kwargs}
    with pytest.raises(ValueError):
        generate_speculative(target, draft, **args)


def test_invalid_model_pairs(models: tuple[SUT, QuantizedSUT]) -> None:
    target, draft = models
    for bad_target, bad_draft in (
        (draft, draft),
        (target, target),
        (target, QuantizedSUT(target, bits=4)),
    ):
        with pytest.raises(ValueError):
            generate_speculative(bad_target, bad_draft, [5], max_new_tokens=1, k=4)
    mismatched = QuantizedSUT(target, bits=8)
    mismatched.config = replace(target.config, context_length=64)
    with pytest.raises(ValueError):
        generate_speculative(target, mismatched, [5], max_new_tokens=1, k=4)


def test_analytic_threshold_has_legal_negative_result() -> None:
    bench = runpy.run_path(str(ROOT / "scripts/spec_bench.py"))
    negative = bench["break_even"](4, 2.0, 2.0, 1.0)
    assert negative["feasible"] is False
    assert negative["strict_positive_gain_alpha_infimum"] is None
    assert negative["ideal_max_speedup"] == 0.5
    # At k=1, C/T=1.5 requires 1+alpha>1.5.
    positive = bench["break_even"](1, 0.5, 1.0, 1.0)
    assert positive["strict_positive_gain_alpha_infimum"] == pytest.approx(0.5)
    assert bench["break_even"](1, 1.0, 1.0, 1.0)["feasible"] is False
    with pytest.raises(ValueError):
        bench["break_even"](1, float("nan"), 1.0, 1.0)


def test_recipe_rejects_relaxed_gate_and_missing_counterbalance(tmp_path: Path) -> None:
    bench = runpy.run_path(str(ROOT / "scripts/spec_bench.py"))
    for changed in ({"orders": [["fp32"]]}, {"random": {"seeds": [42]}}, {"k_values": [1, 2]}):
        config = json.loads((ROOT / "configs/p2_spec.json").read_text())
        config.update(changed)
        with pytest.raises(ValueError):
            bench["run_benchmark"](config, runs_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_benchmark_persists_protocol_and_check_is_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism.record import load_run

    bench = runpy.run_path(str(ROOT / "scripts/spec_bench.py"))
    config = json.loads((ROOT / "configs/p2_spec.json").read_text())
    config.update(warmup=0, repeat=1)
    for variable in bench["THREAD_VARIABLES"]:
        monkeypatch.setenv(variable, "1")
    summary = bench["run_benchmark"](
        config, runs_dir=tmp_path / "runs", report_path=tmp_path / "report.md"
    )
    assert summary["correctness"]["matrix_cases"] == 80
    assert len(summary["records"]) == 38
    assert len(summary["comparisons"]) == 16
    for row in summary["records"]:
        run = load_run(row["path"])
        assert run.config["recipe"] == config
        assert run.measurement.config_hash == row["measurement"]["config_hash"]
        assert len(run.measurement.samples_ms) == 1
        assert run.config["workload"]["eos_id"] is None
    for case in summary["correctness"]["cases"]:
        result = SpeculativeResult(
            case["token_ids"],
            tuple(
                speculative.SpeculativeStep(
                    **{
                        **step,
                        "candidates": tuple(step["candidates"]),
                        "targets": tuple(step["targets"]),
                        "emitted": tuple(step["emitted"]),
                    }
                )
                for step in case["steps"]
            ),
        )
        audit(result, case["prompt_ids"], case["k"])

    def snapshot() -> dict[str, str]:
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    checked = bench["run_benchmark"](
        config, check=True, runs_dir=tmp_path / "runs", report_path=tmp_path / "report.md"
    )
    assert checked["implementation_passed"] and all(r["path"] is None for r in checked["records"])
    assert snapshot() == before


def test_gate_stops_before_measurement_or_persistence(
    models: tuple[SUT, QuantizedSUT], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected test double divergence must abort; the genuine draft stays unchanged."""
    bench = runpy.run_path(str(ROOT / "scripts/spec_bench.py"))
    calls = []

    def divergent(*args: Any, **kwargs: Any) -> SpeculativeResult:
        calls.append(1)
        return SpeculativeResult([0], ())

    monkeypatch.setattr(speculative, "generate_speculative", divergent)
    config = json.loads((ROOT / "configs/p2_spec.json").read_text())
    for variable in bench["THREAD_VARIABLES"]:
        monkeypatch.setenv(variable, "1")
    with pytest.raises(RuntimeError, match="HARD FAIL"):
        bench["run_benchmark"](
            config, runs_dir=tmp_path / "runs", report_path=tmp_path / "report.md"
        )
    assert calls == [1]
    assert not list(tmp_path.iterdir())

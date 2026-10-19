"""Measure CPU weight-only quantization; --check runs the recipe without publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p2_quant.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported schemes and invalid workload sizes before measurement."""
    expected = {
        "task",
        "phase",
        "sut_path",
        "device",
        "threads",
        "quantize_embedding",
        "schemes",
        "group_size",
        "orders",
        "warmup",
        "repeat",
        "prompt_ids",
        "decode_tokens",
        "random",
        "calibration",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("invalid P2-T1 recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P2-T1", "P2", "cpu"):
        raise ValueError("only P2-T1 / P2 / cpu is supported")
    if config["quantize_embedding"] is not False:
        raise ValueError("quantize_embedding must be false; embedding and tied head stay fp32")
    if config["schemes"] != {"fp32": None, "int8": 8, "int4": 4}:
        raise ValueError("expected fp32/int8/int4 schemes")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("requires one thread")
    if config["orders"] != [["fp32", "int8", "int4"], ["int4", "int8", "fp32"]]:
        raise ValueError("requires forward and reverse scheme orders")
    for name, minimum in (("warmup", 0), ("repeat", 1), ("group_size", 1), ("decode_tokens", 1)):
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("sut_path must be nonempty text")
    prompt = config["prompt_ids"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
    ):
        raise ValueError("invalid prompt_ids")
    random = config["random"]
    if not isinstance(random, dict) or set(random) != {
        "seed",
        "count",
        "prompt_length",
        "new_tokens",
    }:
        raise ValueError("invalid random prompt recipe")
    for name, minimum in (("seed", 0), ("count", 1), ("prompt_length", 1), ("new_tokens", 1)):
        if type(random[name]) is not int or random[name] < minimum:
            raise ValueError(f"random.{name} must be an integer >= {minimum}")
    if config["calibration"] != {"report": "results/p0/calibration.md", "resolution_ms": 0.092584}:
        raise ValueError("must cite the existing T4 empirical reference")


def sequence_metrics(expected: list[int], actual: list[int], prompt_length: int) -> dict[str, Any]:
    """Compare generated suffixes; divergence indices are zero-based absolute positions."""
    reference = expected[prompt_length:]
    candidate = actual[prompt_length:]
    count = max(len(reference), len(candidate))
    matches = sum(a == b for a, b in zip(reference, candidate, strict=False))
    first = next(
        (
            i
            for i in range(count)
            if i >= min(len(reference), len(candidate)) or reference[i] != candidate[i]
        ),
        None,
    )
    return {
        "matched_generated_tokens": matches,
        "generated_tokens": count,
        "token_agreement": matches / count if count else 1.0,
        "first_divergence_generated_index": first,
        "first_divergence_absolute_index": None if first is None else first + prompt_length,
        "token_ids": actual,
    }


def teacher_forced_diagnostics(
    sut: SUT, tokens: list[int], prompt_length: int, oracle: SUT | None = None
) -> dict[str, Any]:
    """Locate cached/uncached argmax changes on identical prefixes, including rollover.

    A dequantized fp32 oracle checks implementation equality, independently of
    accuracy degradation. Cached/uncached tolerances concern float arithmetic
    only; an argmax change within tolerance is diagnostic, never a quality gate.
    """
    import numpy as np

    from prism.kvcache import _forward_cached

    cache = None
    oracle_cache = None
    maximum = 0.0
    disagreements = []
    for end in range(prompt_length, len(tokens)):
        window = np.array([tokens[max(0, end - sut.config.context_length) : end]], dtype=np.int64)
        rebuild = cache is None or cache[0][0].shape[2] == sut.config.context_length
        incoming = window if rebuild else window[:, -1:]
        logits, cache = _forward_cached(sut, incoming, None if rebuild else cache)
        naive = sut.forward(window)[0, -1]
        actual = logits[0, -1]
        if not np.isfinite(logits).all() or not np.isfinite(naive).all():
            raise RuntimeError("non-finite inference output")
        if oracle is not None:
            independent, oracle_cache = _forward_cached(
                oracle, incoming, None if rebuild else oracle_cache
            )
            if not np.array_equal(logits, independent):
                raise RuntimeError("quantized Linear disagrees with dequantized fp32 oracle")
        error = float(np.max(np.abs(actual - naive)))
        maximum = max(maximum, error)
        if not np.allclose(actual, naive, atol=3e-5, rtol=3e-5):
            raise RuntimeError(f"cached/uncached logits implementation mismatch at prefix {end}")
        if int(actual.argmax()) != int(naive.argmax()):
            top = np.sort(actual)[-2:]
            disagreements.append(
                {
                    "absolute_index": end,
                    "prefix_ids": tokens[:end],
                    "cached_argmax": int(actual.argmax()),
                    "uncached_argmax": int(naive.argmax()),
                    "cached_top2_margin": float(top[-1] - top[-2]),
                    "max_abs_logit_error": error,
                }
            )
    return {"max_abs_cached_vs_uncached": maximum, "argmax_disagreements": disagreements}


def verify_correctness(
    sut: SUT, models: dict[str, SUT], path: Path, recipe: dict[str, Any]
) -> dict[str, Any]:
    """Check baseline integrity and implementation; describe accuracy without thresholds."""
    import numpy as np

    from prism.quant import QuantizedSUT
    from prism.sut import SUT, check_sut

    baseline = check_sut(path)
    fixture = json.loads((path / "reference_greedy.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(recipe["seed"])
    prompts = rng.integers(
        0, sut.config.vocab_size, size=(recipe["count"], recipe["prompt_length"])
    ).tolist()
    references = [
        sut.generate(p, max_new_tokens=recipe["new_tokens"], use_cache=True) for p in prompts
    ]
    results = {}
    with np.load(path / "reference_logits.npz", allow_pickle=False) as archive:
        tokens = archive["token_ids"][None, :]
        expected_logits = archive["logits"]
    original = sut.forward(tokens)
    for name, model in models.items():
        oracle = None
        ledger = {
            "total_bytes": sum(a.nbytes for a in sut.arrays.values()),
            "compression_ratio": 1.0,
        }
        layer_errors = {}
        if isinstance(model, QuantizedSUT):
            reconstructed = {key: weight.dequantize() for key, weight in model.weights.items()}
            oracle = SUT(model.config, model.arrays | reconstructed, model.tokenizer)
            ledger = model.memory_ledger()
            for key, value in reconstructed.items():
                error = value.astype(np.float64) - sut.arrays[key]
                layer_errors[key] = {
                    "max_abs_error": float(np.abs(error).max()),
                    "mean_squared_error": float(np.mean(error * error)),
                }
        cases = {}
        for case_name, case in fixture["cases"].items():
            generated = model.generate(
                case["prompt_ids"],
                max_new_tokens=case["max_new_tokens"],
                eos_id=case["eos_id"],
                use_cache=True,
            )
            cases[case_name] = sequence_metrics(
                case["token_ids"], generated, len(case["prompt_ids"])
            )
            cases[case_name]["teacher_forced"] = teacher_forced_diagnostics(
                model, generated, len(case["prompt_ids"]), oracle
            )
        logits = model.forward(tokens)[0]
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite reference logits")
        error = logits.astype(np.float64) - expected_logits
        random_results = []
        for prompt, reference in zip(prompts, references, strict=True):
            actual = model.generate(prompt, max_new_tokens=recipe["new_tokens"], use_cache=True)
            item = sequence_metrics(reference, actual, len(prompt))
            item["prompt_ids"] = prompt
            item["teacher_forced"] = teacher_forced_diagnostics(model, actual, len(prompt), oracle)
            random_results.append(item)
        results[name] = {
            "fixture_cases": cases,
            "logits": {
                "max_abs_error": float(np.abs(error).max()),
                "mean_squared_error": float(np.mean(error * error)),
            },
            "random": {
                "seed": recipe["seed"],
                "cases": random_results,
                "token_agreement": sum(r["matched_generated_tokens"] for r in random_results)
                / sum(r["generated_tokens"] for r in random_results),
            },
            "memory": ledger,
            "linear_weight_errors": layer_errors,
        }
    if not np.array_equal(original, sut.forward(tokens)):
        raise RuntimeError("fp32 baseline changed after quantization")
    return {
        "implementation_passed": True,
        "accuracy_gate": None,
        "baseline": baseline,
        "schemes": results,
    }


def run_benchmark(config: dict[str, Any], *, check: bool = False) -> dict[str, Any]:
    """Measure prefill, isolated teacher-forced decode, and Linear component probes."""
    validate_config(config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config["threads"])
    # Set native thread limits before importing NumPy or prism's numerical modules.
    import numpy as np

    from prism.kvcache import _forward_cached
    from prism.measure import measure
    from prism.quant import QuantizedSUT
    from prism.record import load_run, save_run
    from prism.sut import SUT, load_sut

    path = ROOT / config["sut_path"]
    sut = load_sut(path)
    prompt = config["prompt_ids"]
    if (
        max(prompt) >= sut.config.vocab_size
        or len(prompt) + config["decode_tokens"] > sut.config.context_length
        or config["random"]["prompt_length"] > sut.config.context_length
    ):
        raise ValueError("recipe must fit vocabulary/context; timing excludes rollover")
    models: dict[str, SUT] = {
        name: sut
        if bits is None
        else QuantizedSUT(
            sut,
            bits=bits,
            group_size=config["group_size"],
            quantize_embedding=config["quantize_embedding"],
        )
        for name, bits in config["schemes"].items()
    }
    correctness = verify_correctness(sut, models, path, config["random"])
    continuation = sut.generate(prompt, max_new_tokens=config["decode_tokens"], use_cache=True)[
        len(prompt) :
    ]
    prefill_tokens = np.array([prompt], dtype=np.int64)
    runtime = {
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor() or platform.machine(),
        "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
    }
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/prism/quant.py",
            "src/prism/sut.py",
            "src/prism/kvcache.py",
            "scripts/quant_bench.py",
        )
    }
    session = datetime.now(UTC).strftime("%H%M%S%f")
    records = []
    comparisons = []
    for batch, order in enumerate(config["orders"], 1):
        for scheme in order:
            model = models[scheme]
            _, initial_cache = _forward_cached(model, prefill_tokens)
            for phase in ("prefill", "decode"):

                def full(
                    phase: str = phase,
                    model: SUT = model,
                    initial_cache: Any = initial_cache,
                ) -> Any:
                    if phase == "prefill":
                        return _forward_cached(model, prefill_tokens)
                    cache = initial_cache
                    for token in continuation:
                        logits, cache = _forward_cached(
                            model, np.array([[token]], dtype=np.int64), cache
                        )
                    return logits, cache

                # Capture actual Linear inputs on the identical teacher-forced path.
                activations: list[tuple[Any, str]] = []

                class Capture(SUT):
                    def _linear(
                        self,
                        x: Any,
                        name: str,
                        model: SUT = model,
                        activations: list[tuple[Any, str]] = activations,
                    ) -> Any:
                        activations.append((x.copy(), name))
                        return model._linear(x, name)

                capture = Capture(model.config, model.arrays, model.tokenizer)
                if phase == "prefill":
                    _forward_cached(capture, prefill_tokens)
                else:
                    cache = initial_cache
                    for token in continuation:
                        _, cache = _forward_cached(
                            capture, np.array([[token]], dtype=np.int64), cache
                        )
                names = sorted({name for _, name in activations})
                weights = {
                    name: model.weights[f"{name}.weight"].dequantize()
                    if isinstance(model, QuantizedSUT)
                    else model.arrays[f"{name}.weight"]
                    for name in names
                }

                def dequantize(
                    model: SUT = model, activations: list[tuple[Any, str]] = activations
                ) -> None:
                    if isinstance(model, QuantizedSUT):
                        for _, name in activations:
                            model.weights[f"{name}.weight"].dequantize()

                def matmul(
                    model: SUT = model,
                    activations: list[tuple[Any, str]] = activations,
                    weights: dict[str, Any] = weights,
                ) -> None:
                    for x, name in activations:
                        _ = x @ weights[name].T + model.arrays[f"{name}.bias"]

                probes: dict[str, Callable[[], Any]] = {
                    "full": full,
                    "dequantize": dequantize,
                    "matmul_bias": matmul,
                }
                for component, fn in probes.items():
                    record_config = {
                        "task": "P2-T1",
                        "phase": "P2",
                        "recipe": config,
                        "runtime": runtime,
                        "sources_sha256": sources,
                        "manifest_sha256": hashlib.sha256(
                            (path / "MANIFEST.json").read_bytes()
                        ).hexdigest(),
                        "sut_config": asdict(sut.config),
                        "batch": batch,
                        "order": order,
                        "workload": {
                            "scheme": scheme,
                            "phase": phase,
                            "component": component,
                            "batch_size": 1,
                            "decode_tokens": continuation,
                            "timing_scope": "prefill only"
                            if phase == "prefill"
                            else f"{len(continuation)} cached steps; prefill excluded",
                            "linear_calls": len(activations),
                        },
                    }
                    if component == "full":
                        output, cache_output = fn()
                        if not np.isfinite(output).all() or not all(
                            np.isfinite(value).all() for pair in cache_output for value in pair
                        ):
                            raise RuntimeError("non-finite recipe output")
                    measurement = measure(
                        fn, warmup=config["warmup"], repeat=config["repeat"], config=record_config
                    )
                    run_path = None
                    if not check:
                        saved = save_run(
                            ROOT / "runs",
                            f"p2-quant-{session}-b{batch}-{scheme}-{phase}-{component}",
                            record_config,
                            measurement,
                        )
                        measurement = load_run(saved).measurement
                        run_path = str(saved.relative_to(ROOT))
                    records.append(
                        {
                            "batch": batch,
                            "scheme": scheme,
                            "phase": phase,
                            "component": component,
                            "path": run_path,
                            "measurement": asdict(measurement),
                        }
                    )
        for phase in ("prefill", "decode"):
            arms = {
                r["scheme"]: r["measurement"]
                for r in records
                if r["batch"] == batch and r["phase"] == phase and r["component"] == "full"
            }
            for scheme in ("int8", "int4"):
                delta = arms["fp32"]["p50"] - arms[scheme]["p50"]
                comparisons.append(
                    {
                        "batch": batch,
                        "phase": phase,
                        "scheme": scheme,
                        "fp32_minus_quant_p50_ms": delta,
                        "p50_speedup": arms["fp32"]["p50"] / arms[scheme]["p50"],
                        "T4": "不可分辨"
                        if abs(delta) < config["calibration"]["resolution_ms"]
                        else "超过 T4 经验参照；不自动证明显著性",
                    }
                )
    summary = {
        "task": "P2-T1",
        "evidence": "CPU 实测",
        "check_only": check,
        "runtime": runtime,
        "recipe": config,
        "correctness": correctness,
        "records": records,
        "comparisons": comparisons,
        "limitations": (
            "0.5M small model, CPU, fp32 activations/KV/embeddings/tied head; weight-only. "
            "Dequantize on every Linear, then fp32 matmul: no native integer MAC acceleration. "
            "No accuracy acceptance threshold. Implementation passed does not mean "
            "acceptable accuracy. "
            "Decode uses a fixed fp32 teacher-forced continuation and immutable prefill "
            "cache, no rollover; "
            "full decoding includes K/V concatenations. Components replay captured "
            "actual Linear inputs; "
            "dequantize and matmul+bias are separately measured probes, not additive "
            "causal attribution. "
            "Persistent payload ledger excludes Python metadata and transient buffers; "
            "tied head counted once. "
            "All comparison models and probe buffers coexist; tracemalloc peaks are "
            "incremental, miss native "
            "allocations; RSS is process lifetime high-water mark, neither proves model "
            "compression. "
            "Counterbalanced single-process orders, no confidence intervals. T4 floor is "
            "a historical heuristic."
        ),
    }
    if not check:
        first = ROOT / records[0]["path"]
        (first / "comparison.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report = ROOT / "results/p2/quant.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, Any]) -> str:
    """Render full precision/memory/latency tables with per-record provenance."""
    recipe = summary["recipe"]
    rows = [
        "# P2-T1 权重量化报告草稿（待规划层终审）",
        "",
        f"CPU 实测：{summary['runtime']['cpu_model']}；设备 CPU；单线程。",
        f"预热 {recipe['warmup']} 次，重复 {recipe['repeat']} 次；两个相反顺序批次；单位 ms。",
        "实现通过不表示精度可接受；本任务不设精度合格阈值。",
        "",
        "## 精度与精确载荷簿记",
        "",
        "token 一致率只计生成后缀；首分歧为生成后缀零基索引。logits 对比 reference_logits.npz。",
        "",
        "| 方案 | within 一致率/首分歧 | rollover 一致率/首分歧 | "
        "logits max/MSE | 随机一致率 | 模型 bytes | 压缩比 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scheme, result in summary["correctness"]["schemes"].items():
        cases = result["fixture_cases"]
        columns = [
            f"{cases[n]['token_agreement']:.6f}/{cases[n]['first_divergence_generated_index']}"
            for n in ("within_window", "window_rollover")
        ]
        logits = result["logits"]
        memory = result["memory"]
        rows.append(
            f"| {scheme} | {' | '.join(columns)} | "
            f"{logits['max_abs_error']:.9g}/{logits['mean_squared_error']:.9g} | "
            f"{result['random']['token_agreement']:.6f} | {memory['total_bytes']} | "
            f"{memory['compression_ratio']:.6f}× |"
        )
    rows += [
        "",
        f"随机 prompts：种子 {recipe['random']['seed']}，{recipe['random']['count']} 个，"
        f"长度 {recipe['random']['prompt_length']}，生成 {recipe['random']['new_tokens']}；对比 "
        f"cached fp32。",
        "",
        "精确簿记（包含 padding 和 fp32 scales，零点为隐式 0；不计 Python 元数据）：",
        "",
        "```json",
        json.dumps(
            {k: v["memory"] for k, v in summary["correctness"]["schemes"].items()}, indent=2
        ),
        "```",
        "",
        "## 时延与峰值旁证",
        "",
        "prefill 为整段提示；decode 为固定 fp32 前缀下 32 个 cached 步骤，排除 prefill。",
        "每条记录含完整 config hash、env 指纹及原始 samples。",
        "",
        "| batch | 方案 | 阶段 | 成分 | p50 | p95 | p99 | traced peak bytes | RSS bytes | "
        "config hash | env 指纹 | run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary["records"]:
        m = record["measurement"]
        rows.append(
            f"| {record['batch']} | {record['scheme']} | {record['phase']} | "
            f"{record['component']} | {m['p50']:.6f} | {m['p95']:.6f} | {m['p99']:.6f} | "
            f"{m['peak_tracemalloc_bytes']} | {m['max_rss_bytes']} | {m['config_hash']} | "
            f"{m['env_fingerprint']} | {record['path']} |"
        )
    rows += [
        "",
        "## 反量化与矩阵乘占比（独立探针估算）",
        "",
        "来自实际 Linear 输入的重放；分母为完整回路 mean。独立测量并不保证相加为 100%，"
        "matmul 含 bias，输出头仍属 fp32 且不计入此 Linear 探针。",
        "",
        "| batch | 方案 | 阶段 | 反量化/full mean | matmul+bias/full mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in summary["records"]:
        if record["component"] != "full":
            continue
        matched = {
            r["component"]: r["measurement"]["mean"]
            for r in summary["records"]
            if (r["batch"], r["scheme"], r["phase"])
            == (record["batch"], record["scheme"], record["phase"])
        }
        mean = matched["full"]
        dq = "0 (无反量化)" if record["scheme"] == "fp32" else f"{matched['dequantize'] / mean:.2%}"
        rows.append(
            f"| {record['batch']} | {record['scheme']} | {record['phase']} | {dq} | "
            f"{matched['matmul_bias'] / mean:.2%} |"
        )
    rows += [
        "",
        "## T4 经验阈值",
        "",
        "阈值 0.092584 ms，来源 results/p0/calibration.md；"
        "低于该阈值的差异不作为收益证据。超过阈值不自动证明显著性。",
        "",
        "```json",
        json.dumps(summary["comparisons"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 实现、诊断和边界",
        "",
        summary["limitations"],
        "",
        "INT8 每输出通道 [-127,127]；INT4 按输入列分组 [-7,7]，真 nibble 打包，"
        "低 nibble 在先；fp32 scales，ties-to-even，尾组补 0。embedding 与共享头保留 fp32。",
        "逐层权重误差、随机逐用例结果、cached/非 cached 的 teacher-forced "
        "前缀定位见首个 run 的 comparison.json。",
        "",
        "复现：`uv run --locked python scripts/quant_bench.py`；"
        "`--check` 校验并执行完整配方，仅打印 JSON，不写盘。",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    """Publish formal evidence, or print checks and measurements without writing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run_benchmark(config, check=args.check), ensure_ascii=False, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()

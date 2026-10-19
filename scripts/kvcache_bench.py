"""Check exact cached decoding and persist counterbalanced P0-protocol timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_kvcache.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def validate_config(config: dict[str, Any]) -> None:
    """Require the approved CPU, batch-one, two-order decode recipe."""
    keys = {"task", "phase", "sut_path", "device", "threads", "calibration", "orders", "decode"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f"recipe requires exactly {sorted(keys)}")
    if (config["task"], config["phase"], config["device"]) != ("P1-T1", "P1", "cpu"):
        raise ValueError("only P1-T1 / P1 / cpu is supported")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("benchmark requires one thread")
    if config["orders"] != [["naive", "cached"], ["cached", "naive"]]:
        raise ValueError("orders must be naive-cached, then cached-naive")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("sut_path must be nonempty text")
    decode = config["decode"]
    if not isinstance(decode, dict) or set(decode) != {
        "prompt_ids",
        "max_new_tokens",
        "warmup",
        "repeat",
    }:
        raise ValueError("invalid decode recipe")
    for key, minimum in (("warmup", 0), ("repeat", 1), ("max_new_tokens", 1)):
        if type(decode[key]) is not int or decode[key] < minimum:
            raise ValueError(f"decode.{key} must be an integer >= {minimum}")
    if decode["max_new_tokens"] != 32:
        raise ValueError("benchmark requires 32 generated tokens")
    prompt = decode["prompt_ids"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
    ):
        raise ValueError("prompt_ids must contain nonnegative integer IDs")
    calibration = config["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "report",
        "resolution_ms",
        "rationale",
    }:
        raise ValueError("invalid calibration reference")
    if (
        calibration["report"] != "results/p0/calibration.md"
        or type(calibration["resolution_ms"]) not in (int, float)
        or calibration["resolution_ms"] != 0.092584
        or not isinstance(calibration["rationale"], str)
        or not calibration["rationale"]
    ):
        raise ValueError("calibration must cite the T4 empirical reference")


def verify_correctness(sut: SUT, sut_path: Path) -> dict[str, Any]:
    """Fail at the first token disagreement; record logit errors without thresholds."""
    import numpy as np

    from prism.kvcache import _forward_cached, _generate_cached
    from prism.sut import check_sut

    check_sut(sut_path)  # Reuse P0 integrity and naive-fixture checks unchanged.
    fixture = json.loads((sut_path / "reference_greedy.json").read_text(encoding="utf-8"))
    cases = {}
    with np.load(sut_path / "reference_logits.npz", allow_pickle=False) as archive:
        reference_logits = archive["logits"]
    for name, case in fixture["cases"].items():
        prompt = case["prompt_ids"]
        kwargs = {"max_new_tokens": case["max_new_tokens"], "eos_id": case["eos_id"]}
        result = _generate_cached(sut, prompt, **kwargs)
        if (
            result.token_ids != case["token_ids"]
            or sut.generate(prompt, **kwargs, use_cache=True) != case["token_ids"]
        ):
            raise RuntimeError(f"cached greedy token mismatch: {name}; stop and investigate")
        expected_rebuilds = {"within_window": 0, "window_rollover": 14}[name]
        if result.cache_rebuilds != expected_rebuilds:
            raise RuntimeError(f"unexpected rebuild count: {name}: {result.cache_rebuilds}")
        cache = None
        max_error = 0.0
        fixture_error = 0.0
        for end in range(len(prompt), len(result.token_ids)):
            window = np.array(
                [result.token_ids[max(0, end - sut.config.context_length) : end]], dtype=np.int64
            )
            if cache is None or cache[0][0].shape[2] == sut.config.context_length:
                logits, cache = _forward_cached(sut, window)
            else:
                logits, cache = _forward_cached(sut, window[:, -1:], cache)
            actual = logits[0, -1]
            naive = sut.forward(window)[0, -1]
            if int(actual.argmax()) != result.token_ids[end] or int(naive.argmax()) != int(
                actual.argmax()
            ):
                raise RuntimeError(f"token mismatch: {name} position {end}; stop and investigate")
            max_error = max(max_error, float(np.max(np.abs(actual - naive))))
            if name == "within_window":
                fixture_error = max(
                    fixture_error, float(np.max(np.abs(actual - reference_logits[end - 1])))
                )
        cases[name] = {
            "matched_tokens": len(result.token_ids),
            "token_ids": result.token_ids,
            "cache_rebuilds": result.cache_rebuilds,
            "cache_bytes": result.cache_bytes,
            "max_abs_logit_error_vs_naive": max_error,
            "max_abs_logit_error_vs_fixture": fixture_error if name == "within_window" else None,
        }
    # Inspect actual owning K/V payloads at a full window, outside the timed region.
    _, cache = _forward_cached(sut, np.zeros((1, sut.config.context_length), dtype=np.int64))
    analytic = (
        sut.config.layers
        * sut.config.heads
        * sut.config.context_length
        * (sut.config.width // sut.config.heads)
        * 2
        * np.dtype(np.float32).itemsize
    )
    actual_bytes = sum(array.nbytes for pair in cache for array in pair)
    if actual_bytes != analytic:
        raise RuntimeError("cache payload disagrees with the float32 analytical estimate")
    return {
        "cases": cases,
        "full_window_analytic_bytes": analytic,
        "full_window_array_bytes": actual_bytes,
        "memory_scope": "K/V owning array payloads; excludes metadata and temporary buffers",
    }


def run_benchmark(config: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Measure fresh-cache calls, saving each of four arms using save_run()."""
    validate_config(config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config["threads"])
    # NumPy and its native runtime must only load after setting thread limits.
    from prism.kvcache import _generate_cached
    from prism.measure import measure
    from prism.record import load_run, save_run
    from prism.sut import load_sut

    sut_path = ROOT / config["sut_path"]
    sut = load_sut(sut_path)
    correctness = verify_correctness(sut, sut_path)
    decode = config["decode"]
    kwargs = {"max_new_tokens": decode["max_new_tokens"], "eos_id": None}
    expected = sut.generate(decode["prompt_ids"], **kwargs)
    cached = _generate_cached(sut, decode["prompt_ids"], **kwargs)
    if expected != cached.token_ids:
        raise RuntimeError("recipe tokens differ between cached and naive; stop and investigate")
    runtime = {
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
    }
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/prism/sut.py", "src/prism/kvcache.py", "scripts/kvcache_bench.py")
    }
    sut_info = {
        "config": asdict(sut.config),
        "manifest_sha256": hashlib.sha256((sut_path / "MANIFEST.json").read_bytes()).hexdigest(),
    }
    session = datetime.now(UTC).strftime("%H%M%S%f")
    comparisons = []
    paths = []
    for batch, order in enumerate(config["orders"], start=1):
        arms = {}
        for arm in order:
            record_config = {
                "task": config["task"],
                "phase": config["phase"],
                "recipe": config,
                "sut": sut_info,
                "runtime": runtime,
                "sources_sha256": sources,
                "batch": batch,
                "order": order,
                "workload": {
                    "arm": arm,
                    "use_cache": arm == "cached",
                    "batch_size": 1,
                    "eos_id": None,
                    "timing_scope": "prompt prefill plus 32 generated tokens",
                    "cache_scope": "fresh cache per call, released before return",
                },
            }
            measurement = measure(
                lambda arm=arm: sut.generate(
                    decode["prompt_ids"], **kwargs, use_cache=arm == "cached"
                ),
                warmup=decode["warmup"],
                repeat=decode["repeat"],
                config=record_config,
            )
            path = save_run(
                runs_dir, f"p1-kvcache-{session}-b{batch}-{arm}", record_config, measurement
            )
            run = load_run(path)  # Summaries come from validated persisted measurements.
            paths.append(path)
            arms[arm] = {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "p50_ms": run.measurement.p50,
                "p95_ms": run.measurement.p95,
                "p99_ms": run.measurement.p99,
                "peak_tracemalloc_bytes": run.measurement.peak_tracemalloc_bytes,
                "max_rss_bytes": run.measurement.max_rss_bytes,
                "config_hash": run.measurement.config_hash,
                "env_fingerprint": run.measurement.env_fingerprint,
            }
            print(f"saved {path.name}", flush=True)
        delta = arms["naive"]["p50_ms"] - arms["cached"]["p50_ms"]
        comparisons.append(
            {
                "batch": batch,
                "order": order,
                "arms": arms,
                "p50_speedup": arms["naive"]["p50_ms"] / arms["cached"]["p50_ms"],
                "p50_naive_minus_cached_ms": delta,
                "difference_label": "不可分辨"
                if abs(delta) < config["calibration"]["resolution_ms"]
                else "超过 T4 经验参照；不自动证明显著性",
                "peak_cached_minus_naive_bytes": arms["cached"]["peak_tracemalloc_bytes"]
                - arms["naive"]["peak_tracemalloc_bytes"],
            }
        )
    summary = {
        "task": "P1-T1",
        "evidence": "CPU 实测",
        "runtime": runtime,
        "warmup": decode["warmup"],
        "repeat": decode["repeat"],
        "calibration": config["calibration"],
        "correctness": correctness,
        "recipe_cached_diagnostics": asdict(cached),
        "comparisons": comparisons,
        "limitations": (
            "Counterbalanced sequential batches in one process; no confidence intervals. "
            "Timings include tracemalloc perturbation and prefill, not isolated per-token latency. "
            "Peak differences include attention/logit/MLP buffers, Python metadata, concatenation "
            "temporaries and sample bookkeeping; they are not cache payload size. "
            "Tracemalloc misses some native allocations; RSS is a process lifetime peak. "
            "After rollover, every step rebuilds; this recipe does not measure that cost."
        ),
    }
    (paths[0] / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run the checked-in recipe; every invocation publishes four new run records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_benchmark(config, ROOT / "runs"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

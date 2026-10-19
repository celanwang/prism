"""Check exact paged decoding and save bounded K/V memory/throughput scans.

--check runs correctness gates before the full configured measurement recipe.
LRU and sliding block traces are explicitly memory simulations, never generation
throughput. Allocation timings isolate reserve/release management, not attention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.paged import PageStats
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_paged.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def validate_config(config: dict[str, Any]) -> None:
    """Validate the CPU recipe before any artifact is written."""
    keys = {
        "task",
        "phase",
        "sut_path",
        "device",
        "threads",
        "block_sizes",
        "budgets_bytes",
        "policies",
        "orders",
        "calibration",
        "decode",
        "allocation",
        "simulation",
    }
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("invalid recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P1-T2", "P1", "cpu"):
        raise ValueError("only P1-T2 / P1 / cpu is supported")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("one CPU thread required")
    if config["orders"] != [["naive", "continuous", "paged"], ["paged", "continuous", "naive"]]:
        raise ValueError("orders must counterbalance all three adapters")
    if config["policies"] != ["sliding", "lru"]:
        raise ValueError("policies must include sliding and simulation-only lru")
    for key in ("block_sizes", "budgets_bytes"):
        values = config[key]
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 1 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must contain unique positive integers")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("invalid sut_path")
    for name, extras in (
        ("decode", {"prompt_ids", "max_new_tokens"}),
        ("allocation", set()),
        ("simulation", {"tokens", "passes"}),
    ):
        section = config[name]
        if not isinstance(section, dict) or set(section) != {"warmup", "repeat"} | extras:
            raise ValueError(f"invalid {name} section")
        for key in {"warmup", "repeat"} | (extras - {"prompt_ids"}):
            minimum = 0 if key == "warmup" else 1
            if type(section[key]) is not int or section[key] < minimum:
                raise ValueError(f"invalid {name}.{key}")
    prompt = config["decode"]["prompt_ids"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
        or config["decode"]["max_new_tokens"] != 32
    ):
        raise ValueError("decode requires a prompt and 32 new tokens")
    calibration = config["calibration"]
    if (
        not isinstance(calibration, dict)
        or set(calibration) != {"report", "resolution_ms", "rationale"}
        or calibration["report"] != "results/p0/calibration.md"
        or type(calibration["resolution_ms"]) not in (int, float)
        or calibration["resolution_ms"] != 0.092584
        or not isinstance(calibration["rationale"], str)
        or not calibration["rationale"]
    ):
        raise ValueError("calibration must cite the P0 T4 empirical reference")


def check_case(sut: SUT, case: dict[str, Any], arm: str, *, block_size: int = 8) -> dict[str, Any]:
    """Drive three adapters through the same independent per-position oracle.

    Token equality is mandatory. Float logit error is diagnostic only. Production
    generate functions must also match this independently stepped sequence.
    """
    import numpy as np

    from prism.kvcache import _forward_cached, _generate_cached
    from prism.paged import PagedKVCache, forward_paged, generate_paged

    if arm not in ("naive", "continuous", "paged"):
        raise ValueError("unknown adapter")
    budget = (
        math.ceil(sut.config.context_length / block_size)
        * block_size
        * (2 * sut.config.layers * sut.config.width * 4)
    )
    paged = PagedKVCache(sut.config, block_size=block_size, budget_bytes=budget)
    continuous = None
    tokens = list(case["prompt_ids"])
    rebuilds = 0
    max_error = 0.0
    for _ in range(case["max_new_tokens"]):
        window = np.array([tokens[-sut.config.context_length :]], dtype=np.int64)
        oracle = sut.forward(window)[0, -1]
        if arm == "naive":
            actual = oracle
        elif arm == "continuous":
            if continuous is None or continuous[0][0].shape[2] == sut.config.context_length:
                rebuilds += int(continuous is not None)
                continuous = None
                logits, continuous = _forward_cached(sut, window)
            else:
                logits, continuous = _forward_cached(sut, window[:, -1:], continuous)
            actual = logits[0, -1]
        else:
            if paged.length == 0 or paged.length == sut.config.context_length:
                rebuilds += int(paged.length > 0)
                paged.clear(eviction=True)
                logits = forward_paged(sut, window, paged)
            else:
                logits = forward_paged(sut, window[:, -1:], paged)
            actual = logits[0, -1]
        expected = int(oracle.argmax())
        if int(actual.argmax()) != expected:
            raise RuntimeError(f"{arm} disagrees with naive at position {len(tokens)}")
        if expected != case["token_ids"][len(tokens)]:
            raise RuntimeError(f"{arm} disagrees with fixture at position {len(tokens)}")
        error = float(np.max(np.abs(actual - oracle)))
        if not np.isfinite(error):
            raise RuntimeError("nonfinite logit diagnostic")
        max_error = max(max_error, error)
        tokens.append(expected)
        if expected == case["eos_id"]:
            break
    kwargs = {"max_new_tokens": case["max_new_tokens"], "eos_id": case["eos_id"]}
    if arm == "paged":
        result = generate_paged(
            sut, case["prompt_ids"], **kwargs, block_size=block_size, budget_bytes=budget
        )
        production, actual_rebuilds = result.token_ids, result.cache_rebuilds
    elif arm == "continuous":
        result = _generate_cached(sut, case["prompt_ids"], **kwargs)
        production, actual_rebuilds = result.token_ids, result.cache_rebuilds
    else:
        production = sut.generate(case["prompt_ids"], **kwargs)
        actual_rebuilds = 0
    if tokens != case["token_ids"] or production != tokens or actual_rebuilds != rebuilds:
        raise RuntimeError(f"{arm} production generation mismatch")
    return {
        "matched_tokens": len(tokens),
        "cache_rebuilds": rebuilds,
        "max_abs_logit_error_vs_naive": max_error,
    }


def verify_correctness(sut: SUT, sut_path: Path, block_sizes: list[int]) -> dict[str, Any]:
    """Check fixtures and random boundary prompts before collecting any timings."""
    import numpy as np

    from prism.paged import generate_paged
    from prism.sut import check_sut

    check_sut(sut_path)
    fixture = json.loads((sut_path / "reference_greedy.json").read_text())
    cases = {}
    for name, case in fixture["cases"].items():
        adapters = {}
        for arm in ("naive", "continuous", "paged"):
            for block in block_sizes if arm == "paged" else [block_sizes[0]]:
                row = check_case(sut, case, arm, block_size=block)
                expected = 14 if name == "window_rollover" and arm != "naive" else 0
                if row["cache_rebuilds"] != expected:
                    raise RuntimeError("unexpected fixture rebuild count")
                adapters[f"{arm}-b{block}" if arm == "paged" else arm] = row
        cases[name] = adapters
    random_cases = 0
    for seed in (42, 20261019):
        for length in (1, 7, 18, 64, 126, 127, 128, 129, 257):
            prompt = (
                np.random.default_rng(seed).integers(sut.config.vocab_size, size=length).tolist()
            )
            expected = sut.generate(prompt, max_new_tokens=8)
            if sut.generate(prompt, max_new_tokens=8, use_cache=True) != expected:
                raise RuntimeError("continuous random prompt mismatch")
            for block in block_sizes:
                budget = (
                    math.ceil(sut.config.context_length / block)
                    * block
                    * (2 * sut.config.layers * sut.config.width * 4)
                )
                actual = generate_paged(
                    sut, prompt, max_new_tokens=8, block_size=block, budget_bytes=budget
                )
                if actual.token_ids != expected:
                    raise RuntimeError("paged random prompt mismatch")
                random_cases += 1
    return {"cases": cases, "random_prompt_paged_comparisons": random_cases}


def allocation_cycle(sut: SUT, block: int, budget: int, length: int) -> PageStats:
    """Time lazy page reservation and full release, with no model forward or gather."""
    from prism.paged import PagedKVCache

    cache = PagedKVCache(sut.config, block_size=block, budget_bytes=budget)
    if length:
        cache.reserve(length)
    stats = cache.stats
    cache.clear()
    return stats


def simulate(sut: SUT, block: int, budget: int, policy: str, tokens: int, passes: int) -> PageStats:
    """Exercise a fixed synthetic trace; cold/missed payloads reconstruct exactly."""
    import numpy as np

    from prism.paged import PagedKVCache

    cache = PagedKVCache(sut.config, block_size=block, budget_bytes=budget, policy=policy)
    count = math.ceil(tokens / block)
    c = sut.config

    def payload(logical: int) -> np.ndarray:
        used = min(block, tokens - logical * block)
        return np.full(
            (2, c.layers, 1, c.heads, used, c.width // c.heads), logical, dtype=np.float32
        )

    if cache.capacity:
        for _ in range(passes):
            for logical in range(count):
                for request in (logical, 0):
                    cache.access_block(request, min(block, tokens - request * block), payload)
    return cache.stats


def run_benchmark(config: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Save allocation scans, simulated eviction scans, and real decode comparisons."""
    validate_config(config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config["threads"])
    # Load NumPy only after setting thread environment in the entry-point process.
    from prism.kvcache import _generate_cached
    from prism.measure import measure
    from prism.paged import generate_paged
    from prism.record import load_run, save_run
    from prism.sut import load_sut

    sut_path = ROOT / config["sut_path"]
    sut = load_sut(sut_path)
    correctness = verify_correctness(sut, sut_path, config["block_sizes"])
    decode = config["decode"]
    prompt = decode["prompt_ids"]
    kwargs = {"max_new_tokens": decode["max_new_tokens"], "eos_id": None}
    expected = sut.generate(prompt, **kwargs)
    continuous = _generate_cached(sut, prompt, **kwargs)
    if continuous.token_ids != expected:
        raise RuntimeError("recipe continuous mismatch")
    runtime = {
        "device": "cpu",
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
    }
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/prism/sut.py",
            "src/prism/kvcache.py",
            "src/prism/paged.py",
            "scripts/paged_bench.py",
        )
    }
    common = {
        "task": "P1-T2",
        "phase": "P1",
        "recipe": config,
        "runtime": runtime,
        "sources_sha256": sources,
        "sut": {
            "config": asdict(sut.config),
            "manifest_sha256": hashlib.sha256(
                (sut_path / "MANIFEST.json").read_bytes()
            ).hexdigest(),
        },
    }
    session = datetime.now(UTC).strftime("%H%M%S%f")
    paths = []

    def record(
        slug: str, workload: dict[str, Any], fn: Callable[[], Any], protocol: dict[str, Any]
    ) -> dict[str, Any]:
        record_config = {**common, "workload": workload}
        measurement = measure(
            fn, warmup=protocol["warmup"], repeat=protocol["repeat"], config=record_config
        )
        path = save_run(runs_dir, f"p1-paged-{session}-{slug}", record_config, measurement)
        run = load_run(path)
        paths.append(path)
        result = asdict(run.measurement)
        result.pop("samples_ms")
        result["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        print(f"saved {path.name}", flush=True)
        return result

    positions = min(sut.config.context_length, len(prompt) + decode["max_new_tokens"] - 1)
    per_token = 2 * sut.config.layers * sut.config.width * 4
    points = []
    for block in config["block_sizes"]:
        for budget in config["budgets_bytes"]:
            required = math.ceil(positions / block) * block * per_token
            feasible = budget // (block * per_token) >= math.ceil(positions / block)
            live = min(positions, (budget // (block * per_token)) * block)
            stats = allocation_cycle(sut, block, budget, live)
            analytic = stats.resident_pages * block * per_token
            if stats.allocated_bytes != analytic or stats.allocated_bytes > budget:
                raise RuntimeError("actual page payload disagrees with bounded analytical model")
            row = {
                "block_size": block,
                "budget_bytes": budget,
                "required_bytes": required,
                "status": "feasible" if feasible else "infeasible",
                "reason": None
                if feasible
                else "K/V pool cannot retain the complete correct window",
                "allocation_probe_tokens": live,
                "allocation_stats": asdict(stats),
                "allocation_fragmentation": stats.fragmentation,
                "analytic_bytes": analytic,
                "actual_minus_analytic_bytes": 0,
            }
            row["allocation_measurement"] = record(
                f"b{block}-m{budget}-allocation",
                {
                    "kind": "allocation",
                    "block_size": block,
                    "budget_bytes": budget,
                    "tokens": live,
                    "generation_status": row["status"],
                    "timing_scope": "construct, reserve pages, snapshot stats, release all pages",
                },
                lambda b=block, m=budget, n=live: allocation_cycle(sut, b, m, n),
                config["allocation"],
            )
            row["simulations"] = {}
            for policy in config["policies"]:
                sim = simulate(
                    sut,
                    block,
                    budget,
                    policy,
                    config["simulation"]["tokens"],
                    config["simulation"]["passes"],
                )
                measurement = record(
                    f"b{block}-m{budget}-{policy}-simulation",
                    {
                        "kind": "memory_simulation",
                        "policy": policy,
                        "block_size": block,
                        "budget_bytes": budget,
                        "trace": (
                            "for each pass and each logical block: request block, request block 0"
                        ),
                        "payload": "float32 fill with logical block ID; reconstruct on each miss",
                        "generation_comparison": False,
                    },
                    lambda b=block, m=budget, p=policy: simulate(
                        sut, b, m, p, config["simulation"]["tokens"], config["simulation"]["passes"]
                    ),
                    config["simulation"],
                )
                row["simulations"][policy] = {
                    "evidence": "memory simulation only",
                    "stats": asdict(sim),
                    "hit_rate": sim.hit_rate,
                    "fragmentation": sim.fragmentation,
                    "measurement": measurement,
                }
            if feasible:
                result = generate_paged(
                    sut, prompt, **kwargs, block_size=block, budget_bytes=budget
                )
                if result.token_ids != expected:
                    raise RuntimeError("recipe paged mismatch")
                row["generation_stats"] = asdict(result.stats)
                row["generation_hit_rate"] = result.stats.hit_rate
                row["generation_fragmentation"] = result.stats.fragmentation
                row["cache_rebuilds"] = result.cache_rebuilds
            points.append(row)

    # Fixed T and nested block sizes only: no claim across budgets or changing T.
    fragments = [
        (p["block_size"], p["generation_fragmentation"])
        for p in points
        if p["status"] == "feasible"
    ]
    for small, f_small in fragments:
        for large, f_large in fragments:
            if large % small == 0 and f_large < f_small:
                raise RuntimeError("fixed-T nested-block fragmentation is not monotone")
    comparisons = []
    for batch, order in enumerate(config["orders"], 1):
        arms: dict[str, Any] = {}
        for arm in order:
            if arm == "paged":
                arms[arm] = []
                for point in points:
                    if point["status"] != "feasible":
                        continue
                    b, m = point["block_size"], point["budget_bytes"]
                    measurement = record(
                        f"order{batch}-b{b}-m{m}-decode",
                        {
                            "kind": "generation",
                            "arm": arm,
                            "policy": "sliding",
                            "block_size": b,
                            "budget_bytes": m,
                            "order": order,
                            "batch": batch,
                            "timing_scope": "fresh cache, prompt prefill plus 32 emitted tokens",
                        },
                        lambda b=b, m=m: generate_paged(
                            sut, prompt, **kwargs, block_size=b, budget_bytes=m
                        ),
                        decode,
                    )
                    measurement.update(
                        block_size=b,
                        budget_bytes=m,
                        tokens_per_second_at_p50=32000 / measurement["p50"],
                    )
                    arms[arm].append(measurement)
            else:
                measurement = record(
                    f"order{batch}-{arm}-decode",
                    {
                        "kind": "generation",
                        "arm": arm,
                        "order": order,
                        "batch": batch,
                        "timing_scope": "fresh cache, prompt prefill plus 32 emitted tokens",
                    },
                    lambda a=arm: sut.generate(prompt, **kwargs, use_cache=a == "continuous"),
                    decode,
                )
                measurement["tokens_per_second_at_p50"] = 32000 / measurement["p50"]
                arms[arm] = measurement
        for point in arms["paged"]:
            delta = arms["continuous"]["p50"] - point["p50"]
            point["continuous_over_paged_p50"] = arms["continuous"]["p50"] / point["p50"]
            point["continuous_minus_paged_p50_ms"] = delta
            point["difference_label"] = (
                "不可分辨"
                if abs(delta) < config["calibration"]["resolution_ms"]
                else "超过 T4 经验参照；不自动证明显著性"
            )
        comparisons.append({"batch": batch, "order": order, "arms": arms})
    summary = {
        "task": "P1-T2",
        "runtime": runtime,
        "warmup": decode["warmup"],
        "repeat": decode["repeat"],
        "correctness": correctness,
        "positions": positions,
        "bytes_per_token": per_token,
        "continuous_final_payload_bytes": continuous.cache_bytes,
        "points": points,
        "comparisons": comparisons,
        "limitations": (
            "Decode is CPU measurement; LRU and sliding traces are memory simulations. "
            "Infeasible points have no generation timing or throughput. Pool budgets cover only "
            "owning K/V page arrays, excluding metadata, weights, gather and forward temporaries. "
            "Actual nbytes equals pages * block_size * bytes_per_token; unused budget is not "
            "fragmentation. Allocation microbenchmarks include Python management, not attention. "
            "Single-sequence lazy pages need not save memory versus T1 exact-size arrays. "
            "Tracemalloc peaks are auxiliary and miss some native allocations; RSS is a process "
            "lifetime peak. P0 tracing perturbs timing. Counterbalanced sequential orders share "
            "one process, without confidence intervals. This decode recipe stays within the "
            "window; fixture checks independently exercise full invalidation and rebuilding. "
            "Lookup hit rates include writes and per-layer block gathers; simulation hit rates "
            "refer to the synthetic trace and are not comparable generation hit rates."
        ),
    }
    (paths[0] / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run correctness gates and the checked-in recipe, publishing fresh UTC runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--check", action="store_true", help="require all correctness gates (always on)"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_benchmark(config, ROOT / "runs"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

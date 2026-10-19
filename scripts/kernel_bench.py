"""Measure one operator at a time; --check verifies gates/recipes without writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.ablating import AblatingSUT
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p2_kernels.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
RECIPES = {"qkv": "qkv_split", "attention": "attention_bt", "mlp": "mlp_reuse"}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate all three independent recipes before doing any work."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed on bundled switches, missing workloads and invalid protocols."""
    if not isinstance(config, dict) or set(config) != {
        "task",
        "phase",
        "sut_path",
        "device",
        "threads",
        "calibration",
        "recipes",
    }:
        raise ValueError("invalid P2-T3 recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P2-T3", "P2", "cpu"):
        raise ValueError("requires P2-T3 / P2 / cpu")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("requires one native thread")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("sut_path must be nonempty text")
    calibration = config["calibration"]
    if (
        not isinstance(calibration, dict)
        or set(calibration) != {"report", "resolution_ms", "rationale"}
        or calibration["report"] != "results/p0/calibration.md"
        or calibration["resolution_ms"] != 0.092584
        or not isinstance(calibration["rationale"], str)
        or not calibration["rationale"]
    ):
        raise ValueError("requires the existing T4 empirical reference")
    recipes = config["recipes"]
    if not isinstance(recipes, list) or len(recipes) != 3:
        raise ValueError("requires three independent recipes")
    names = []
    for recipe in recipes:
        if not isinstance(recipe, dict) or set(recipe) != {
            "name",
            "variant",
            "warmup",
            "repeat",
            "orders",
            "prefill",
            "decode",
        }:
            raise ValueError("invalid individual recipe keys")
        name, variant = recipe["name"], recipe["variant"]
        if not isinstance(name, str) or name not in RECIPES or variant != RECIPES[name]:
            raise ValueError("each recipe must enable exactly its own variant")
        names.append(name)
        for key, minimum in (("warmup", 0), ("repeat", 1)):
            if type(recipe[key]) is not int or recipe[key] < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if recipe["orders"] != [["baseline", variant], [variant, "baseline"]]:
            raise ValueError("requires forward and reverse two-arm orders")
        prefill, decode = recipe["prefill"], recipe["decode"]
        if (
            not isinstance(prefill, dict)
            or set(prefill) != {"length", "seed"}
            or type(prefill["length"]) is not int
            or prefill["length"] != 128
            or type(prefill["seed"]) is not int
            or prefill["seed"] < 0
        ):
            raise ValueError("requires prefill T=128 and a nonnegative seed")
        if (
            not isinstance(decode, dict)
            or set(decode) != {"prompt_ids", "tokens"}
            or type(decode["tokens"]) is not int
            or decode["tokens"] != 32
            or not isinstance(decode["prompt_ids"], list)
            or not decode["prompt_ids"]
            or any(type(t) is not int or t < 0 for t in decode["prompt_ids"])
        ):
            raise ValueError("requires a valid prompt and decode 32 tokens")
    if set(names) != set(RECIPES):
        raise ValueError("requires each independent recipe exactly once")


def threshold_verdict(deltas_ms: list[float], resolution_ms: float) -> str:
    """Require both orders to clear T4 in the same direction (baseline − variant).

    Above-threshold repeatability is descriptive evidence, not a significance
    test. A reversed/too-small difference is always reported as indistinguishable.
    """
    if (
        len(deltas_ms) != 2
        or not math.isfinite(resolution_ms)
        or resolution_ms <= 0
        or any(not math.isfinite(d) for d in deltas_ms)
    ):
        raise ValueError("requires two finite deltas and a positive finite T4 threshold")
    if all(d >= resolution_ms for d in deltas_ms):
        return "值得"
    if all(d <= -resolution_ms for d in deltas_ms):
        return "不值得"
    return "不可分辨"


def verify_variants(sut: SUT, path: Path, variants: list[str]) -> dict[str, Any]:
    """Gate every token of both independent fixtures, cached and uncached."""
    from prism.ablating import AblatingSUT
    from prism.sut import check_sut

    baseline = check_sut(path)  # Includes fixture hashes; do not trust edited goldens.
    cases = json.loads((path / "reference_greedy.json").read_text(encoding="utf-8"))["cases"]
    results = {}
    for variant in variants:
        model = AblatingSUT(sut, variant)
        results[variant] = {}
        for name, case in cases.items():
            modes = {}
            for cached in (False, True):
                actual = model.generate(
                    case["prompt_ids"],
                    max_new_tokens=case["max_new_tokens"],
                    eos_id=case["eos_id"],
                    use_cache=cached,
                )
                if actual != case["token_ids"]:
                    raise RuntimeError(f"token gate failed: {variant}/{name}/cached={cached}")
                modes["cached" if cached else "uncached"] = {
                    "passed": True,
                    "matched_generated_tokens": len(actual) - len(case["prompt_ids"]),
                    "token_ids": actual,
                }
            results[variant][name] = modes
    return {"passed": True, "baseline": baseline, "variants": results}


def run_benchmark(
    config: dict[str, Any], *, recipe_name: str, check: bool = False
) -> dict[str, Any]:
    """Publish separate per-arm/workload/order runs, never a combined optimization."""
    validate_config(config)
    if recipe_name not in RECIPES:
        raise ValueError("unknown recipe")
    for key in THREAD_VARIABLES:
        os.environ[key] = str(config["threads"])
    # Numerical modules must load after native thread limits are set.
    import numpy as np

    from prism.ablating import AblatingSUT
    from prism.measure import measure
    from prism.record import load_run, save_run
    from prism.sut import load_sut

    recipes = [r for r in config["recipes"] if r["name"] == recipe_name]
    path = ROOT / config["sut_path"]
    sut = load_sut(path)
    for recipe in recipes:
        decode = recipe["decode"]
        if (
            recipe["prefill"]["length"] > sut.config.context_length
            or len(decode["prompt_ids"]) + decode["tokens"] > sut.config.context_length
            or max(decode["prompt_ids"]) >= sut.config.vocab_size
        ):
            raise ValueError("timed workloads must fit vocabulary/context without rollover")
    correctness = verify_variants(sut, path, ["baseline", *(r["variant"] for r in recipes)])
    runtime = {
        "device": "cpu",
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor() or platform.machine(),
        "thread_environment": {k: os.environ[k] for k in THREAD_VARIABLES},
    }
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/prism/ablating.py",
            "src/prism/sut.py",
            "src/prism/kvcache.py",
            "src/prism/measure.py",
            "scripts/kernel_bench.py",
        )
    }
    session = datetime.now(UTC).strftime("%H%M%S%f")
    records = []
    conclusions = []
    for recipe in recipes:
        variant = recipe["variant"]
        rng = np.random.default_rng(recipe["prefill"]["seed"])
        prefill = rng.integers(0, sut.config.vocab_size, size=(1, 128), dtype=np.int64)
        prompt = recipe["decode"]["prompt_ids"]
        # Fixed, common teacher-forced continuation: exclude sampling divergence.
        continuation = sut.generate(
            prompt, max_new_tokens=recipe["decode"]["tokens"], use_cache=True
        )[len(prompt) :]
        incoming = [np.array([[t]], dtype=np.int64) for t in continuation]
        prompt_array = np.array([prompt], dtype=np.int64)
        for batch, order in enumerate(recipe["orders"], 1):
            for workload in ("prefill", "decode"):
                for arm in order:
                    model = AblatingSUT(sut, arm)
                    _, initial_cache = model.forward_cached(prompt_array)

                    def fn(
                        model: AblatingSUT = model,
                        workload: str = workload,
                        initial_cache: Any = initial_cache,
                        prefill: Any = prefill,
                        incoming: Any = incoming,
                    ) -> Any:
                        if workload == "prefill":
                            return model.forward_cached(prefill)
                        cache = initial_cache
                        for tokens in incoming:
                            logits, cache = model.forward_cached(tokens, cache)
                        return logits, cache

                    output, _ = fn()
                    if not np.isfinite(output).all():
                        raise RuntimeError("non-finite benchmark output")
                    record_config = {
                        "task": "P2-T3",
                        "phase": "P2",
                        # Only this recipe is embedded; no other enabled ablation.
                        "recipe": recipe,
                        "calibration": config["calibration"],
                        "sut_path": config["sut_path"],
                        "sut_config": asdict(sut.config),
                        "manifest_sha256": hashlib.sha256(
                            (path / "MANIFEST.json").read_bytes()
                        ).hexdigest(),
                        "runtime": runtime,
                        "sources_sha256": sources,
                        "batch": batch,
                        "order": order,
                        "workload": {
                            "name": workload,
                            "arm": arm,
                            "batch_size": 1,
                            "prefill_ids": prefill[0].tolist() if workload == "prefill" else prompt,
                            "continuation_ids": [] if workload == "prefill" else continuation,
                            "timing_scope": "T=128 cached prefill including KV creation"
                            if workload == "prefill"
                            else (
                                "32 teacher-forced cached steps; prefill excluded; "
                                "fresh cache reference each sample"
                            ),
                            "scratch_scope": (
                                "last shape retained across layers/calls; "
                                "warmed before timing; logits/cache own storage"
                            ),
                        },
                    }
                    m = measure(
                        fn, warmup=recipe["warmup"], repeat=recipe["repeat"], config=record_config
                    )
                    saved = None
                    if not check:
                        saved = save_run(
                            ROOT / "runs",
                            f"p2-kernels-{session}-{recipe['name']}-b{batch}-{arm}-{workload}",
                            record_config,
                            m,
                        )
                        m = load_run(saved).measurement
                    records.append(
                        {
                            "recipe": recipe["name"],
                            "batch": batch,
                            "workload": workload,
                            "arm": arm,
                            "path": None if saved is None else str(saved.relative_to(ROOT)),
                            "measurement": asdict(m),
                        }
                    )
                    print(
                        f"{recipe['name']} b{batch} {workload} {arm}: p50={m.p50:.6f} ms",
                        flush=True,
                    )
        for workload in ("prefill", "decode"):
            deltas = []
            for batch in (1, 2):
                arms = {
                    r["arm"]: r["measurement"]
                    for r in records
                    if r["recipe"] == recipe["name"]
                    and r["batch"] == batch
                    and r["workload"] == workload
                }
                deltas.append(arms["baseline"]["p50"] - arms[variant]["p50"])
            conclusions.append(
                {
                    "recipe": recipe["name"],
                    "workload": workload,
                    "baseline_minus_variant_p50_ms": deltas,
                    "resolution_ms": config["calibration"]["resolution_ms"],
                    "verdict": threshold_verdict(deltas, config["calibration"]["resolution_ms"]),
                }
            )
    summary = {
        "task": "P2-T3",
        "evidence": "CPU 实测",
        "check": check,
        "runtime": runtime,
        "calibration": config["calibration"],
        "correctness": correctness,
        "records": records,
        "conclusions": conclusions,
        "limitations": (
            "T4 is an empirical reference, not statistical significance. Two sequential "
            "counterbalanced orders in one process; thermal/frequency/scheduling drift remains. "
            "Tracemalloc perturbs allocations, misses some native allocations; RSS is process "
            "lifetime high-water mark. No bundled optimizations or per-token threshold rescaling. "
            "QKV verdict evaluates splitting; reverse its interpretation for merging. Decode is "
            "teacher-forced, excludes prefill, EOS and rollover. Scratch warm allocation is "
            "excluded; layout packing is included."
        ),
    }
    if not check:
        # Each recipe gets its own correctness/data summary in its own run tree.
        first_path = ROOT / records[0]["path"]
        (first_path / "comparison.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    """Run a selected independent recipe, or check all without publishing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "-o", "--output", type=Path, help="JSON report path inside a p2-kernels run"
    )
    args = parser.parse_args()
    if args.output is not None:
        target = args.output.resolve()
        relative = target.relative_to(ROOT / "runs")
        if len(relative.parts) < 2 or "-p2-kernels-" not in relative.parts[0] or args.check:
            parser.error("-o requires a runs/<date>-p2-kernels-*/ path and no --check")
    config = load_config(args.config)
    # Execute each independent recipe separately, with its own saved summary.
    names = [args.recipe] if args.recipe else list(RECIPES)
    summaries = [run_benchmark(config, check=args.check, recipe_name=name) for name in names]
    rendered = json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

"""Calibrate P1 simulation against CPU serial batch-one proxies, never real batching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p1_calibrate.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def validate_config(config: dict[str, Any]) -> None:
    """Fail before timing or writing on unsupported recipes."""
    keys = {
        "task",
        "phase",
        "sut_path",
        "device",
        "threads",
        "sim_recipe",
        "prefill",
        "decode",
        "calibration",
    }
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("invalid P1-T4 recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P1-T4", "P1", "cpu"):
        raise ValueError("requires P1-T4 / P1 / cpu")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("requires one native thread")
    for key in ("sut_path", "sim_recipe"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"invalid {key}")
    for section, fields in (
        ("prefill", {"sequence_lengths", "token_id", "warmup", "repeat"}),
        ("decode", {"batch_sizes", "context_tokens", "token_id", "warmup", "repeat"}),
    ):
        values = config[section]
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError(f"invalid {section}")
        for key in fields - {"sequence_lengths", "batch_sizes"}:
            minimum = 0 if key in ("token_id", "warmup") else 1
            if type(values[key]) is not int or values[key] < minimum:
                raise ValueError(f"invalid {section}.{key}")
    if config["prefill"]["sequence_lengths"] != [1, 32, 128]:
        raise ValueError("requires prefill T=1,32,128")
    if config["decode"]["batch_sizes"] != [1, 2, 4, 8]:
        raise ValueError("requires decode B=1,2,4,8")
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
        raise ValueError("requires P0-T4 calibration reference")
    json.dumps(config, allow_nan=False)


def relative_deviation(predicted: float, measured: float) -> float:
    """Signed (simulation - measurement) / measurement; never hide zero denominators."""
    if not math.isfinite(predicted) or not math.isfinite(measured) or measured <= 0:
        raise ValueError("requires finite prediction and positive measured time")
    return (predicted - measured) / measured


def fit_prefill(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit p50 = intercept + slope*T by ordinary least squares, without timing IO."""
    import numpy as np

    x = np.array([point["size"] for point in points], dtype=float)
    y = np.array([point["p50"] for point in points], dtype=float)
    if len(x) != 3 or list(x) != [1, 32, 128] or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("prefill fit requires three positive T=1,32,128 points")
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("nonpositive fitted slope; stop rather than invent a model")
    return {
        "method": "OLS p50_ms = intercept_ms + slope_ms_per_token * T",
        "slope_ms_per_token": float(slope),
        "intercept_ms": float(intercept),
        "residuals_ms": (y - (intercept + slope * x)).tolist(),
        "simulator_mapping": "Only slope is representable; intercept omitted, not absorbed.",
    }


def make_decode_proxy(sut: Any, batch: int, context: int, token_id: int) -> Any:
    """Construct B independent caches once; each timed call advances each exactly once.

    T1's forward never mutates past caches. Each call reuses the same fixed-context
    past, drops its newly produced cache/logits, and selects one greedy token per
    sequence. Setup/prefill is excluded; no threading, vectorized batch, or B divisor.
    """
    import numpy as np

    from prism.kvcache import _forward_cached

    states = []
    for index in range(batch):
        prompt = np.full((1, context), (token_id + index) % sut.config.vocab_size, dtype=np.int64)
        logits, cache = _forward_cached(sut, prompt)
        next_token = np.array([[int(logits[0, -1].argmax())]], dtype=np.int64)
        states.append((next_token, cache))

    def step() -> None:
        for token, cache in states:
            logits, present = _forward_cached(sut, token, cache)
            int(logits[0, -1].argmax())
            # Include output disposal per sequence in the serial total.
            del logits, present

    return step


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def compare_simulations(
    original: dict[str, Any], calibrated: dict[str, Any], runs_dir: Path, session: str
) -> list[dict[str, Any]]:
    """Replay the same arrivals at λ=100/s, B=4, with unchanged scheduler semantics."""
    from prism.schedsim import save_simulation, simulate, validate_config

    points = []
    for label, recipe in (("before", original), ("after", calibrated)):
        for policy in ("static", "continuous"):
            point = deepcopy(recipe)
            for key in ("arrival_rates_per_s", "batch_sizes", "policies"):
                point.pop(key)
            point.update(arrival_rate_per_s=100, batch_size=4, policy=policy)
            validate_config(point)
            points.append((label, point))
    rows = []
    for label, point in points:
        result = simulate(point)
        path = save_simulation(
            result, runs_dir, f"p1-calibrate-{session}-{label}-{point['policy']}"
        )
        rows.append(
            {
                "stage": label,
                "policy": point["policy"],
                "path": display_path(path),
                "summary": result.summary,
            }
        )
    return rows


def run_calibration(
    config: dict[str, Any], runs_dir: Path, *, check: bool = False
) -> dict[str, Any]:
    """Persist seven measured points and four simulated comparisons with provenance."""
    validate_config(config)
    config = deepcopy(config)
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    # Load NumPy/BLAS only after setting the thread environment in the CLI process.
    import numpy as np

    from prism.kvcache import _forward_cached
    from prism.measure import measure
    from prism.record import load_run, save_run
    from prism.schedsim import decode_ms
    from prism.sut import check_sut, load_sut

    original_path = ROOT / config["sim_recipe"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if check:
        for section in ("prefill", "decode"):
            config[section].update(warmup=0, repeat=2)
        original["duration_ms"] = min(original["duration_ms"], 500)
    sut_path = ROOT / config["sut_path"]
    check_sut(sut_path)
    sut = load_sut(sut_path)
    if (
        config["decode"]["context_tokens"] >= sut.config.context_length
        or max(config["prefill"]["sequence_lengths"]) > sut.config.context_length
        or any(config[s]["token_id"] >= sut.config.vocab_size for s in ("prefill", "decode"))
    ):
        raise ValueError("recipe exceeds SUT vocabulary/context")
    runtime = {
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor() or platform.machine(),
        "device": "cpu",
        "machine": platform.machine(),
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
    }
    common = {
        "task": "P1-T4",
        "phase": "P1",
        "recipe": config,
        "runtime": runtime,
        "evidence": "smoke only; 非实验证据" if check else "CPU 实测 / serial batch-one proxy",
        "sut": {
            "config": asdict(sut.config),
            "manifest_sha256": hashlib.sha256(
                (sut_path / "MANIFEST.json").read_bytes()
            ).hexdigest(),
        },
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/p1_calibrate.py",
                "src/prism/kvcache.py",
                "src/prism/sut.py",
                "src/prism/measure.py",
                "src/prism/schedsim.py",
            )
        },
        "original_sim_recipe_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
    }
    session = datetime.now(UTC).strftime("%H%M%S%f") + ("-check" if check else "")
    paths = []
    measured: dict[str, list[dict[str, Any]]] = {"prefill": [], "decode": []}
    for kind, sizes in (
        ("prefill", config["prefill"]["sequence_lengths"]),
        ("decode", config["decode"]["batch_sizes"]),
    ):
        protocol = config[kind]
        for size in sizes:
            workload = {
                "kind": kind,
                "size": size,
                "timing_scope": "fresh cached prefill including result disposal"
                if kind == "prefill"
                else (
                    "B independent batch-one sequences, one cached forward and greedy selection "
                    "each, serial total / 1 step; setup excluded; fixed immutable past reused "
                    "per sample"
                ),
                "context_tokens": 0 if kind == "prefill" else protocol["context_tokens"],
            }
            if kind == "prefill":
                tokens = np.full((1, size), protocol["token_id"], dtype=np.int64)

                def fn(tokens=tokens):
                    return _forward_cached(sut, tokens)
            else:
                fn = make_decode_proxy(sut, size, protocol["context_tokens"], protocol["token_id"])
            record_config = {**common, "workload": workload}
            measurement = measure(
                fn, warmup=protocol["warmup"], repeat=protocol["repeat"], config=record_config
            )
            path = save_run(
                runs_dir, f"p1-calibrate-{session}-{kind}-{size}", record_config, measurement
            )
            paths.append(path)
            row = asdict(load_run(path).measurement)
            row.pop("samples_ms")
            row.update(size=size, path=display_path(path))
            measured[kind].append(row)
            print(f"saved {path.name}", flush=True)
    fit = fit_prefill(measured["prefill"])
    calibrated = deepcopy(original)
    calibrated["service_model"] = {
        "prefill_ms_per_token": fit["slope_ms_per_token"],
        "decode_nodes": [[r["size"], r["p50"]] for r in measured["decode"]],
        "source": {
            **common,
            "measurements": measured,
            "prefill_fit": fit,
            "limits": (
                "CPU small model serial batch-one proxy, not real concurrent batching. "
                f"Decode calibrated only at context={config['decode']['context_tokens']}; "
                "prefill intercept omitted; no held-out validation. "
                "No 0.35 scaling assumption used."
            ),
        },
    }
    # Reject noisy, nonmonotonic nodes via the simulator; never smooth silently.
    simulations = compare_simulations(original, calibrated, runs_dir, session)
    deviations = []
    for kind in ("prefill", "decode"):
        for row in measured[kind]:
            size = row["size"]
            before = (
                original["service_model"]["prefill_ms_per_token"] * size
                if kind == "prefill"
                else decode_ms(original["service_model"], size)
            )
            after = (
                fit["slope_ms_per_token"] * size
                if kind == "prefill"
                else decode_ms(calibrated["service_model"], size)
            )
            deviations.append(
                {
                    "kind": kind,
                    "size": size,
                    "measured_p50_ms": row["p50"],
                    "before_ms": before,
                    "after_ms": after,
                    "before_relative_deviation": relative_deviation(before, row["p50"]),
                    "after_relative_deviation": relative_deviation(after, row["p50"]),
                }
            )
    summary = {
        **common,
        "measurements": measured,
        "prefill_fit": fit,
        "deviations": deviations,
        "simulations": simulations,
        "calibrated_recipe": display_path(paths[0] / "calibrated_schedsim.json"),
        "original_recipe": display_path(paths[0] / "original_schedsim.json"),
    }
    write_json(paths[0] / "original_schedsim.json", original)
    write_json(paths[0] / "calibrated_schedsim.json", calibrated)
    write_json(paths[0] / "calibration.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--check", action="store_true", help="tiny persisted smoke recipe; not evidence"
    )
    parser.add_argument(
        "--replay", type=Path, help="replay four simulations from a saved calibration.json"
    )
    args = parser.parse_args(argv)
    if args.replay:
        summary = json.loads(args.replay.read_text(encoding="utf-8"))
        original = json.loads((ROOT / summary["original_recipe"]).read_text(encoding="utf-8"))
        calibrated = json.loads((ROOT / summary["calibrated_recipe"]).read_text(encoding="utf-8"))
        session = datetime.now(UTC).strftime("%H%M%S%f") + "-replay"
        result = compare_simulations(original, calibrated, args.runs_dir, session)
        if [r["summary"] for r in result] != [r["summary"] for r in summary["simulations"]]:
            raise AssertionError("same-config simulation replay differs")
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_calibration(config, args.runs_dir, check=args.check)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

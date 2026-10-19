"""Reproduce loopback deployment gates and measured HTTP / nominal T3 comparisons.

--check is a small real smoke flow, explicitly not experimental evidence.
P0 measure() retains round timings; separate artifacts retain per-request timings.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from prism.serve import InferenceService
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p3_serve.json"


def stats(values: list[float]) -> dict[str, float | None]:
    """Return request quantiles, preserving missing observations as null."""
    import numpy as np

    if not values:
        return {key: None for key in ("p50", "p95", "p99")}
    return dict(
        zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99])), strict=True)
    )


def call(
    port: int,
    body: dict[str, Any],
    *,
    stream: bool = False,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    """Consume JSON/SSE completely; retain client E2E and server queue waiting."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    first_token = None
    try:
        connection.request(
            "POST",
            "/generate/stream" if stream else "/generate",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        tokens = []
        if stream and response.status == 200:
            data: dict[str, Any] = {}
            event = ""
            for line in response:
                if line.startswith(b"event: "):
                    event = line.decode().strip().split(": ", 1)[1]
                elif line.startswith(b"data: "):
                    item = json.loads(line[6:])
                    if event == "token":
                        tokens.append(item["token_id"])
                        if first_token is None:
                            first_token = (time.perf_counter() - start) * 1000
                    elif event == "error":
                        raise RuntimeError(item["error"])
                    elif event == "done":
                        data = item
            if "token_ids" not in data:
                raise RuntimeError("missing SSE done")
            if data["token_ids"][len(body["prompt"]) :] != tokens:
                raise RuntimeError("incomplete SSE tokens")
        else:
            data = json.loads(response.read())
        return {
            "status": response.status,
            "e2e_ms": (time.perf_counter() - start) * 1000,
            "queue_wait_ms": data.get("queue_wait_ms"),
            "inference_ms": data.get("inference_ms"),
            "ttft_ms": first_token,
            "token_ids": data.get("token_ids"),
        }
    finally:
        connection.close()


def equivalence(sut: SUT, service: InferenceService, seed: int) -> dict[str, Any]:
    """Hard fail unless fixture and random full/stream/local tokens all agree."""
    from prism.sut import DEFAULT_SUT_PATH

    fixture = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())["cases"]
    cases = [
        (name, c["prompt_ids"], c["max_new_tokens"], c["eos_id"], c["token_ids"])
        for name, c in fixture.items()
    ]
    rng = random.Random(seed)
    cases += [
        (
            f"random-{length}",
            [rng.randrange(sut.config.vocab_size) for _ in range(length)],
            7,
            None,
            None,
        )
        for length in (1, 23, 127, 140)
    ]
    rows = []
    for name, prompt, count, eos, golden in cases:
        local = sut.generate(prompt, max_new_tokens=count, eos_id=eos, use_cache=True)
        whole = call(service.port, {"prompt": prompt, "max_new_tokens": count, "eos_id": eos})
        stream = call(
            service.port, {"prompt": prompt, "max_new_tokens": count, "eos_id": eos}, stream=True
        )
        if (
            whole["status"] != 200
            or stream["status"] != 200
            or whole["token_ids"] != local
            or stream["token_ids"] != local
            or (golden is not None and golden != local)
        ):
            raise RuntimeError(f"deployment equivalence failed: {name}")
        rows.append({"case": name, "passed": True, "token_ids": local})
    return {"passed": True, "oracle": "independent SUT.generate(use_cache=True)", "cases": rows}


def validate_recipe(recipe: dict[str, Any]) -> None:
    """Fail before starting threads or writing artifacts for unsupported recipes."""
    if recipe["concurrency"] != [1, 2, 4, 8] or recipe["endpoints"] != ["whole", "stream"]:
        raise ValueError("expected concurrency 1,2,4,8 and whole/stream")
    if recipe["device"] != "cpu" or recipe["native_threads"] != 1:
        raise ValueError("only single-native-thread CPU recipe supported")
    for name in (
        "repeat",
        "requests_per_round",
        "check_requests",
        "queue_capacity",
        "http_workers",
    ):
        if type(recipe[name]) is not int or recipe[name] < 1:
            raise ValueError(f"invalid {name}")
    for name in ("seed", "warmup", "http_pending", "max_new_tokens"):
        if type(recipe[name]) is not int or recipe[name] < 0:
            raise ValueError(f"invalid {name}")
    if recipe["http_workers"] < max(recipe["concurrency"]):
        raise ValueError(
            "HTTP workers must cover client concurrency to isolate inference backpressure"
        )
    for name in ("prompt_length", "output_length"):
        spec = recipe[name]
        if (
            spec["kind"] != "uniform"
            or type(spec["min"]) is not int
            or type(spec["max"]) is not int
            or not 1 <= spec["min"] <= spec["max"]
        ):
            raise ValueError("expected positive integer uniform lengths")
    if recipe["max_new_tokens"] < max(recipe["output_length"]["max"], 64):
        raise ValueError("output cap must cover workload and fixtures")


def hardware() -> str:
    """Describe the actual CPU; env.lock retains stable protocol provenance."""
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def run_benchmark(recipe: dict[str, Any], runs_dir: Path, *, check: bool = False) -> dict[str, Any]:
    """Measure fixed waves and persist each grid point through save_run()."""
    from prism.measure import measure
    from prism.record import save_run
    from prism.schedsim import save_simulation, simulate
    from prism.serve import InferenceService
    from prism.sut import load_sut

    validate_recipe(recipe)
    sut = load_sut(ROOT / recipe["sut_path"])
    repeat, warmup = (2, 0) if check else (recipe["repeat"], recipe["warmup"])
    count = recipe["check_requests"] if check else recipe["requests_per_round"]
    rng = random.Random(recipe["seed"])
    workload = []
    for _ in range(count):
        length = rng.randint(recipe["prompt_length"]["min"], recipe["prompt_length"]["max"])
        workload.append(
            {
                "prompt": [rng.randrange(sut.config.vocab_size) for _ in range(length)],
                "max_new_tokens": rng.randint(
                    recipe["output_length"]["min"], recipe["output_length"]["max"]
                ),
            }
        )
    identity = uuid4().hex[:8]
    result: dict[str, Any] = {
        "rows": [],
        "hardware_device": hardware() + " / CPU / 1 native thread",
    }
    with InferenceService(
        sut,
        queue_capacity=recipe["queue_capacity"],
        http_workers=recipe["http_workers"],
        http_pending=recipe["http_pending"],
        max_new_tokens=recipe["max_new_tokens"],
    ) as service:
        result["equivalence"] = equivalence(sut, service, recipe["seed"])
        for endpoint in recipe["endpoints"]:
            for concurrency in recipe["concurrency"]:
                config = {
                    **recipe,
                    "endpoint": endpoint,
                    "client_concurrency": concurrency,
                    "warmup": warmup,
                    "repeat": repeat,
                    "requests_per_round": count,
                    "hardware_device": result["hardware_device"],
                    "evidence": "smoke 非实验证据"
                    if check
                    else "实测 loopback CPU engineering evidence",
                    "measurement_unit": "complete client round (not request latency)",
                }
                records: list[dict[str, Any]] = []
                round_number = 0
                with ThreadPoolExecutor(max_workers=concurrency) as clients:

                    def run_round(
                        concurrency: int = concurrency,
                        endpoint: str = endpoint,
                        records: list[dict[str, Any]] = records,
                    ) -> None:
                        nonlocal round_number
                        for offset in range(0, count, concurrency):
                            wave = workload[offset : offset + concurrency]
                            barrier = threading.Barrier(len(wave))
                            futures = [
                                clients.submit(
                                    call,
                                    service.port,
                                    body,
                                    stream=endpoint == "stream",
                                    barrier=barrier,
                                )
                                for body in wave
                            ]
                            for index, future in enumerate(futures, offset):
                                row = future.result()
                                if row["status"] not in (200, 429):
                                    raise RuntimeError(f"unexpected HTTP status: {row['status']}")
                                row.pop("token_ids")
                                if round_number >= warmup:
                                    records.append(
                                        {"round": round_number - warmup, "request_id": index, **row}
                                    )
                        round_number += 1

                    measurement = measure(run_round, warmup=warmup, repeat=repeat, config=config)
                path = save_run(
                    runs_dir, f"p3-serve-{identity}-{endpoint}-c{concurrency}", config, measurement
                )
                completed = [r for r in records if r["status"] == 200]
                summary = {
                    "mode": "实测",
                    "concurrency": concurrency,
                    "endpoint": endpoint,
                    "requests": len(records),
                    "completed": len(completed),
                    "429": sum(r["status"] == 429 for r in records),
                    "503": 0,
                    "request_e2e_ms": stats([r["e2e_ms"] for r in completed]),
                    "request_queue_wait_ms": stats([r["queue_wait_ms"] for r in completed]),
                    "request_ttft_ms": stats(
                        [r["ttft_ms"] for r in completed if r["ttft_ms"] is not None]
                    ),
                    "rejection_e2e_ms": stats([r["e2e_ms"] for r in records if r["status"] == 429]),
                    "round_measurement_ms": {
                        k: getattr(measurement, k) for k in ("p50", "p95", "p99")
                    },
                    "config_hash": measurement.config_hash,
                    "env_fingerprint": measurement.env_fingerprint,
                }
                sim_config = json.loads((ROOT / recipe["sim_recipe"]).read_text())
                sim_config.update(
                    task="P3-T1",
                    phase="P3",
                    policy="static",
                    batch_size=1,
                    batch_timeout_ms=0,
                    seed=recipe["seed"],
                    queue_capacity=recipe["queue_capacity"],
                    queue_timeout_ms=None,
                    duration_ms=sum(measurement.samples_ms),
                    arrival_rate_per_s=len(records) / (sum(measurement.samples_ms) / 1000),
                    prompt_length=recipe["prompt_length"],
                    output_length=recipe["output_length"],
                    comparison=recipe["semantics"]["comparison"],
                    evidence="smoke 非实验证据" if check else "模拟 nominal-parameter comparison",
                )
                simulation = simulate(sim_config)
                simulation_path = save_simulation(
                    simulation, runs_dir, f"p3-serve-{identity}-{endpoint}-c{concurrency}-sim"
                )
                summary["simulation"] = simulation.summary
                for name, data in (
                    ("requests.json", records),
                    ("summary.json", summary),
                    ("workload.json", workload),
                    ("equivalence.json", result["equivalence"]),
                ):
                    (path / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
                result["rows"].append(
                    {**summary, "path": str(path), "simulation_path": str(simulation_path)}
                )
    report_path = Path(result["rows"][0]["path"]) / "REPORT.md"
    report_path.write_text(report(result, warmup, repeat, check), encoding="utf-8")
    result["report"] = str(report_path)
    return result


def report(result: dict[str, Any], warmup: int, repeat: int, check: bool) -> str:
    """Keep measured/request/round/simulation evidence clearly distinguished."""
    lines = [
        "# P3-T1 HTTP 服务与部署等价性",
        "",
        f"硬件/设备：{result['hardware_device']}；"
        f"每个 grid point 预热 {warmup} 轮、重复 {repeat} 轮。",
        "smoke 非实验证据。"
        if check
        else "实测：本机 loopback、同进程客户端与服务、小模型；不代表生产性能。",
        "分位数均用 NumPy linear；请求级只统计成功请求（含排队），拒绝单独记录。",
        "Measurement 是完整一轮耗时，不能解读为单请求时延。",
        "",
        "硬门槛：fixture within_window/window_rollover + 固定种子随机 prompt，",
        "HTTP 整包 == SSE done/逐 token 重建 == 独立本地 generate(use_cache=True)。",
        "完整序列见 equivalence.json；事件门控真流式验证见 tests/test_serve.py。",
        "",
        "| 模式 | C | 成功/请求 | 429 | 请求 E2E p50/p95/p99 ms | "
        "等待 p50/p95/p99 ms | 整轮 p50/p95/p99 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def quantiles(value: dict[str, Any]) -> str:
        return "/".join(
            "null" if value[k] is None else f"{value[k]:.3f}" for k in ("p50", "p95", "p99")
        )

    for row in result["rows"]:
        lines.append(
            f"| 实测 {row['endpoint']} | {row['concurrency']} | "
            f"{row['completed']}/{row['requests']} | "
            f"{row['429']} | {quantiles(row['request_e2e_ms'])} | "
            f"{quantiles(row['request_queue_wait_ms'])} | "
            f"{quantiles(row['round_measurement_ms'])} |"
        )
        sim = row["simulation"]
        lines.append(
            f"| 模拟 {row['endpoint']} | {row['concurrency']} | "
            f"{sim['completed']}/{sim['arrivals']} | "
            f"{sim['rejected']} | {quantiles(sim['latency_ms'])} | "
            f"{quantiles(sim['wait_ms'])} | 无 |"
        )
    lines += [
        "",
        "T3 同名义参数：batch_size=1、batch_timeout_ms=0、相同 Q 与长度分布。",
        "模拟的 offered rate 为实测每轮请求数/整轮时长；模拟 Poisson 到达，实测固定序列分波闭环。",
        "T3 保留 P0 无缓存服务成本代理（来源、哈希、近似均在模拟 config.json）；",
        "未标定 HTTP/cached 成本，未建模 socket/线程/客户端与 tracemalloc 开销，",
        "长度与服务耗时相关性也不同。仅做分布对照，不比较逐请求轨迹或声称拟合精度。",
        "高并发拒绝数受 OS/线程调度影响；固定种子固定的是请求序列和并发模式。",
        "等待仅含服务推理队列；TTFT、拒绝 E2E 和每请求明细见 summary.json/requests.json。",
        "",
    ]
    for row in result["rows"]:
        lines += [
            f"- `{row['path']}`",
            f"  env `{row['env_fingerprint']}`；config `{row['config_hash']}`；",
            f"  模拟 `{row['simulation_path']}`。",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Run the recipe and print artifact pointers, optionally writing a report (-o)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    # Native libraries read these at first import; NumPy imports stay inside functions.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    result = run_benchmark(json.loads(args.config.read_text()), args.runs_dir, check=args.check)
    if args.output:
        args.output.write_text(Path(result["report"]).read_text(), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

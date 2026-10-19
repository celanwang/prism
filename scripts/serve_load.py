"""Seeded open-loop HTTP load grid and calibrated serial T3 comparison.

--check exercises real HTTP with a tiny recipe; it is never experimental evidence.
-o writes the run report. Original P1 calibration is pinned and snapshotted.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import queue
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p3_load.json"


def stats(values: list[float]) -> dict:
    import numpy as np

    return (
        dict(
            zip(
                ("p50", "p95", "p99"),
                map(float, np.percentile(values, [50, 95, 99], method="linear")),
                strict=True,
            )
        )
        if values
        else dict.fromkeys(("p50", "p95", "p99"))
    )


def hardware() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate before model loading, threads or artifacts; pin calibration bytes."""
    from prism.schedsim import validate_config

    json.dumps(recipe, allow_nan=False)
    if recipe["device"] != "cpu" or recipe["native_threads"] != 1:
        raise ValueError("only one-native-thread CPU recipe supported")
    for name in ("duration_ms", "client_timeout_s", "server_socket_timeout_s", "queue_sample_ms"):
        value = recipe[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {name}")
    for name in ("warmup", "seed", "http_pending", "repeat", "http_workers"):
        if type(recipe[name]) is not int or recipe[name] < (name in ("repeat", "http_workers")):
            raise ValueError(f"invalid {name}")
    for name in ("arrival_rates_per_s", "concurrency", "queue_capacities"):
        values = recipe[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid grid {name}")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
            if name != "arrival_rates_per_s" and type(value) is not int:
                raise ValueError(f"integer required for {name}")
    if recipe["http_workers"] <= max(recipe["concurrency"]):
        raise ValueError("HTTP workers must exceed C to isolate inference backpressure")
    source = ROOT / recipe["calibrated_recipe"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != recipe["calibrated_recipe_sha256"]:
        raise ValueError("calibration SHA-256 mismatch")
    calibrated = json.loads(source.read_text())
    if calibrated["service_model"]["source"].get("task") != "P1-T4":
        raise ValueError("expected P1-T4 calibrated service model")
    config = simulation_config(recipe, calibrated, recipe["arrival_rates_per_s"][0], 2)
    validate_config(config)
    if recipe["client_timeout_s"] * 1000 <= recipe["p99_threshold_ms"]:
        raise ValueError("timeout must exceed p99 threshold to avoid censoring the gate")
    return calibrated


def simulation_config(recipe: dict, calibrated: dict, rate: float, capacity: int) -> dict:
    return {
        **calibrated,
        "task": "P3-T2",
        "phase": "P3",
        "policy": "static",
        "batch_size": 1,
        "batch_timeout_ms": 0,
        "seed": recipe["seed"],
        "duration_ms": recipe["duration_ms"],
        "arrival_rate_per_s": rate,
        "queue_capacity": capacity,
        "queue_timeout_ms": None,
        "prompt_length": recipe["prompt_length"],
        "output_length": recipe["output_length"],
        "p99_threshold_ms": recipe["p99_threshold_ms"],
        "queue_growth": recipe["queue_growth"],
    }


def make_workload(config: dict, vocab_size: int) -> list[dict]:
    """Exactly T3's seeded arrival/length RNG; prompt contents use another stream."""
    import numpy as np

    from prism.schedsim import sample_length

    rng = np.random.default_rng(config["seed"])
    tokens = np.random.default_rng(config["seed"] + 1)
    workload = []
    arrival = 0.0
    while True:
        arrival += float(rng.exponential(1000 / config["arrival_rate_per_s"]))
        if arrival >= config["duration_ms"]:
            break
        prompt = sample_length(rng, config["prompt_length"])
        output = sample_length(rng, config["output_length"])
        workload.append(
            {
                "id": len(workload),
                "arrival_ms": arrival,
                "prompt": tokens.integers(0, vocab_size, size=prompt).tolist(),
                "max_new_tokens": output,
            }
        )
    return workload


def avalanche(latency: dict, series: list[dict], config: dict) -> dict:
    """T3's exact OR rule on a piecewise-constant queue trace (sampled for HTTP)."""
    horizon = config["duration_ms"]
    areas = [0.0] * 4
    area = 0.0
    for left, right in zip(series, [*series[1:], {"time_ms": horizon}], strict=True):
        start, end = min(left["time_ms"], horizon), min(right["time_ms"], horizon)
        area += max(0, end - start) * left["queue_length"]
        for i in range(4):
            a = horizon / 2 + i * horizon / 8
            b = a + horizon / 8
            areas[i] += max(0, min(end, b) - max(start, a)) * left["queue_length"]
    means = [value / (horizon / 8) for value in areas]
    slope = (means[-1] - means[0]) / (3 * horizon / 8 / 1000)
    final = next(r["queue_length"] for r in reversed(series) if r["time_ms"] <= horizon)
    growing = (
        all(b > a for a, b in zip(means[:-1], means[1:], strict=True))
        and slope > config["queue_growth"]["min_slope_per_s"]
        and final >= config["queue_growth"]["min_queue"]
    )
    reasons = []
    if latency["p99"] is not None and latency["p99"] > config["p99_threshold_ms"]:
        reasons.append("p99_threshold")
    if growing:
        reasons.append("finite_window_queue_growth_proxy")
    return {
        "avalanche": bool(reasons),
        "avalanche_reasons": reasons,
        "late_window_mean_queues": means,
        "queue_slope_per_s": slope,
        "horizon_queue": final,
        "mean_queue_length": area / horizon,
        "max_queue_length": max(r["queue_length"] for r in series),
    }


def serial_simulation(workload: list[dict], config: dict, concurrency: int | None) -> dict:
    """Batch-one T3 FIFO replay with a bounded, ideal zero-transport client pool.

    C slots last until full response or client timeout; inference continues to drain
    after timeout, as whole-response SUT.generate cannot be interrupted mid-call.
    """
    from prism.schedsim import decode_ms

    waiting: deque[dict] = deque()
    active = None
    slots: list[float] = []
    series = [{"time_ms": 0.0, "queue_length": 0}]
    records = []
    model = config["service_model"]
    timeout_ms = config.get("client_timeout_s", float("inf")) * 1000

    def start(row: dict, now: float) -> dict:
        cost = len(row["prompt"]) * model["prefill_ms_per_token"] + row[
            "max_new_tokens"
        ] * decode_ms(model, 1)
        row.update(start_ms=now, finish_ms=now + cost, queue_wait_ms=now - row["arrival_ms"])
        # Active and waiting client deadlines are fixed independently of server work.
        return row

    def complete_until(now: float) -> None:
        nonlocal active
        while active is not None and active["finish_ms"] <= now:
            end = active["finish_ms"]
            active["e2e_ms"] = min(end - active["arrival_ms"], timeout_ms)
            active["status"] = 200 if end <= active["arrival_ms"] + timeout_ms else "timeout"
            if active["status"] == 200:
                slots.remove(active["arrival_ms"])
            if waiting:
                active = start(waiting.popleft(), end)
            else:
                active = None
            series.append({"time_ms": end, "queue_length": len(waiting)})

    for request in workload:
        now = request["arrival_ms"]
        complete_until(now)
        slots[:] = [t for t in slots if t + timeout_ms > now]
        row = {**request}
        records.append(row)
        if concurrency is not None and len(slots) >= concurrency:
            row.update(status="client_full", finish_ms=now)
        elif active is not None and len(waiting) >= config["queue_capacity"]:
            row.update(status=429, finish_ms=now, e2e_ms=0)
        else:
            slots.append(now)
            if active is None:
                active = start(row, now)
            else:
                waiting.append(row)
        series.append({"time_ms": now, "queue_length": len(waiting)})
    complete_until(float("inf"))
    summary = summarize(records, config["duration_ms"])
    summary.update(avalanche(summary["latency_ms"], series, config))
    return {"summary": summary, "requests": records, "queue_series": series}


def summarize(records: list[dict], horizon: float) -> dict:
    completed = [r for r in records if r["status"] == 200]
    end = max([horizon, *(r.get("finish_ms", horizon) for r in records)])
    return {
        "arrivals": len(records),
        "completed": len(completed),
        "rejected": sum(r["status"] == 429 for r in records),
        "timed_out": sum(r["status"] == "timeout" for r in records),
        "client_full": sum(r["status"] == "client_full" for r in records),
        "http_503": sum(r["status"] == 503 for r in records),
        "transport_errors": sum(r["status"] == "transport_error" for r in records),
        "latency_ms": stats([r["e2e_ms"] for r in completed]),
        "wait_ms": stats([r["queue_wait_ms"] for r in completed]),
        "throughput_requests_per_s": sum(r["finish_ms"] <= horizon for r in completed)
        / (horizon / 1000),
        "drained_throughput_requests_per_s": len(completed) / (end / 1000),
        "drained_until_ms": end,
    }


def http_round(service: Any, workload: list[dict], config: dict, concurrency: int) -> dict:
    """No hidden client backlog: reserve an idle worker or record client_full."""
    records: list[dict] = []
    series = [{"time_ms": 0.0, "queue_length": 0}]
    idle = queue.Queue()
    for i in range(concurrency):
        idle.put(i)
    stop = threading.Event()
    ready = threading.Barrier(concurrency + 1)
    # Prestart executor workers outside the arrival clock.
    start = 0.0

    def wait_ready() -> None:
        ready.wait()

    def sample() -> None:
        while not stop.is_set():
            series.append(
                {
                    "time_ms": (time.perf_counter() - start) * 1000,
                    "queue_length": service.queue.qsize(),
                }
            )
            stop.wait(config["queue_sample_ms"] / 1000)

    def call(request: dict, slot: int) -> dict:
        began = time.perf_counter()
        cpu = time.thread_time()
        row = {**request, "dispatch_ms": (began - start) * 1000 - request["arrival_ms"]}
        connection = http.client.HTTPConnection(
            "127.0.0.1", service.port, timeout=config["client_timeout_s"]
        )
        try:
            connection.request(
                "POST",
                "/generate",
                json.dumps(
                    {"prompt": request["prompt"], "max_new_tokens": request["max_new_tokens"]}
                ),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            data = json.loads(response.read())
            row.update(status=response.status, queue_wait_ms=data.get("queue_wait_ms"))
            if response.status == 200:
                if len(data["token_ids"]) != len(request["prompt"]) + request["max_new_tokens"]:
                    raise RuntimeError("incomplete generated response")
                row["inference_ms"] = data["inference_ms"]
            elif response.status not in (429, 503):
                raise RuntimeError(f"unexpected HTTP status {response.status}")
        except TimeoutError:
            row["status"] = "timeout"
        except (OSError, http.client.HTTPException) as error:
            row.update(status="transport_error", error=type(error).__name__)
        finally:
            connection.close()
            row["finish_ms"] = (time.perf_counter() - start) * 1000
            row["wire_ms"] = (time.perf_counter() - began) * 1000
            row["e2e_ms"] = row["finish_ms"] - request["arrival_ms"]
            row["client_cpu_ms"] = (time.thread_time() - cpu) * 1000
            idle.put(slot)
        return row

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="load-client") as clients:
        initial = [clients.submit(wait_ready) for _ in range(concurrency)]
        ready.wait()
        for future in initial:
            future.result()
        start = time.perf_counter()
        process_cpu = time.process_time()
        sampler = threading.Thread(target=sample)
        sampler.start()
        futures = []
        try:
            for request in workload:
                delay = start + request["arrival_ms"] / 1000 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                try:
                    slot = idle.get_nowait()
                except queue.Empty:
                    records.append({**request, "status": "client_full"})
                else:
                    futures.append(clients.submit(call, request, slot))
            delay = start + config["duration_ms"] / 1000 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            records.extend(future.result() for future in futures)
        finally:
            stop.set()
            sampler.join()
        cpu_ms = (time.process_time() - process_cpu) * 1000
    # A timed-out whole-response call may still be executing in SUT.generate.
    # Isolate rounds by waiting for server jobs too; never cancel their work silently.
    while True:
        with service.lock:
            pending = bool(service.jobs)
        if not pending:
            break
        time.sleep(0.001)
    records.sort(key=lambda r: r["id"])
    summary = summarize(records, config["duration_ms"])
    summary.update(avalanche(summary["latency_ms"], series, config))
    called = [r for r in records if r["status"] != "client_full"]
    completed = [r for r in called if r["status"] == 200]
    summary.update(
        dispatch_ms=stats([r["dispatch_ms"] for r in called]),
        client_cpu_ms=stats([r["client_cpu_ms"] for r in called]),
        transport_residual_ms=stats(
            [r["wire_ms"] - r["queue_wait_ms"] - r["inference_ms"] for r in completed]
        ),
        sampling_gap_ms=stats(
            [b["time_ms"] - a["time_ms"] for a, b in zip(series, series[1:], strict=False)]
        ),
        max_sampling_gap_ms=max(
            [0, *(b["time_ms"] - a["time_ms"] for a, b in zip(series, series[1:], strict=False))]
        ),
        process_cpu_ms=cpu_ms,
    )
    return {"summary": summary, "requests": records, "queue_series": series}


def run_load(recipe: dict, runs_dir: Path, *, check: bool = False) -> dict:
    from prism.env import collect_env, env_fingerprint
    from prism.measure import measure
    from prism.record import save_run
    from prism.schedsim import simulate
    from prism.serve import InferenceService
    from prism.sut import load_sut

    recipe = json.loads(json.dumps(recipe))
    if check:
        recipe.update(recipe["check"])
    calibrated = validate_recipe(recipe)
    sut = load_sut(ROOT / recipe["sut_path"])
    identity = datetime.now(UTC).strftime("%H%M%S%f") + "-" + uuid4().hex[:6]
    root = runs_dir / f"{datetime.now(UTC):%Y%m%d}-p3-load-{identity}{'-check' if check else ''}"
    root.mkdir(parents=True)
    (root / "calibrated_schedsim.json").write_bytes(
        (ROOT / recipe["calibrated_recipe"]).read_bytes()
    )
    write_json(root / "recipe.json", recipe)
    env = collect_env()
    write_json(root / "env.lock", env)
    result = {
        "path": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        "evidence": "smoke 非实验证据" if check else "实测 loopback CPU / calibrated 模拟",
        "hardware_device": hardware() + " / CPU / 1 native thread",
        "env_fingerprint": env_fingerprint(env),
        "config_hash": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        "warmup": recipe["warmup"],
        "repeat": recipe["repeat"],
        "rows": [],
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/prism/serve.py", "src/prism/schedsim.py", "scripts/serve_load.py")
        },
    }
    for rate in recipe["arrival_rates_per_s"]:
        for concurrency in recipe["concurrency"]:
            for capacity in recipe["queue_capacities"]:
                sim_config = simulation_config(recipe, calibrated, rate, capacity)
                workload = make_workload(sim_config, sut.config.vocab_size)
                nominal = simulate(sim_config)
                # Gate the adapter against the original T3 before adding C limits.
                unlimited = serial_simulation(workload, sim_config, None)["summary"]
                for key in ("completed", "rejected", "avalanche", "avalanche_reasons"):
                    if unlimited[key] != nominal.summary[key]:
                        raise RuntimeError(f"T3 adapter differs: {key}")
                for key in ("p50", "p95", "p99"):
                    a, b = unlimited["latency_ms"][key], nominal.summary["latency_ms"][key]
                    if a != b and (a is None or b is None or not math.isclose(a, b, abs_tol=1e-7)):
                        raise RuntimeError("T3 adapter latency differs")
                simulation = serial_simulation(
                    workload,
                    {**sim_config, **{"client_timeout_s": recipe["client_timeout_s"]}},
                    concurrency,
                )
                config = {
                    **recipe,
                    "arrival_rate_per_s": rate,
                    "client_concurrency": concurrency,
                    "queue_capacity": capacity,
                    "hardware_device": result["hardware_device"],
                    "evidence": result["evidence"],
                    "measurement_unit": "whole round including arrival window and drain",
                }
                rounds = []
                index = 0
                with InferenceService(
                    sut,
                    queue_capacity=capacity,
                    http_workers=recipe["http_workers"],
                    http_pending=recipe["http_pending"],
                    max_new_tokens=recipe["output_length"]["max"],
                    socket_timeout=recipe["server_socket_timeout_s"],
                ) as service:

                    def run_round(
                        workload=workload, concurrency=concurrency, rounds=rounds
                    ) -> None:
                        nonlocal index
                        observed = http_round(service, workload, recipe, concurrency)
                        if index >= recipe["warmup"]:
                            rounds.append(observed)
                        index += 1

                    measurement = measure(
                        run_round, warmup=recipe["warmup"], repeat=recipe["repeat"], config=config
                    )
                slug = f"r{rate}-c{concurrency}-q{capacity}"
                point = save_run(root, slug, config, measurement)
                records = [r for trial in rounds for r in trial["requests"]]
                completed = [r for r in records if r["status"] == 200]
                summary = {
                    "rate": rate,
                    "concurrency": concurrency,
                    "queue_capacity": capacity,
                    "config_hash": measurement.config_hash,
                    "env_fingerprint": measurement.env_fingerprint,
                    "point": point.name,
                    "latency_ms": stats([r["e2e_ms"] for r in completed]),
                    "completed": len(completed),
                    "arrivals": len(records),
                    **{
                        key: sum(t["summary"][key] for t in rounds)
                        for key in (
                            "rejected",
                            "timed_out",
                            "client_full",
                            "http_503",
                            "transport_errors",
                        )
                    },
                    "throughput_requests_per_s": sum(
                        t["summary"]["throughput_requests_per_s"] for t in rounds
                    )
                    / len(rounds),
                    "avalanche_rounds": sum(t["summary"]["avalanche"] for t in rounds),
                    "simulation": simulation["summary"],
                }
                predicted = simulation["summary"]["latency_ms"]["p99"]
                measured = summary["latency_ms"]["p99"]
                summary["p99_relative_deviation"] = (
                    (predicted - measured) / measured
                    if predicted is not None and measured is not None and measured > 0
                    else None
                )
                for name, value in (
                    ("workload.json", workload),
                    ("rounds.json", rounds),
                    ("simulation.json", simulation),
                    (
                        "simulation_config.json",
                        {
                            **sim_config,
                            "client_concurrency": concurrency,
                            "client_timeout_s": recipe["client_timeout_s"],
                            "adapter": "serial T3 batch-one FIFO + ideal bounded client pool",
                        },
                    ),
                    ("nominal_t3_summary.json", nominal.summary),
                    ("summary.json", summary),
                ):
                    write_json(point / name, value)
                result["rows"].append(summary)
                print(
                    f"{slug}: successes={summary['completed']}, 429={summary['rejected']}, "
                    f"client_full={summary['client_full']}, avalanche="
                    f"{summary['avalanche_rounds']}/{len(rounds)}",
                    flush=True,
                )
    write_json(root / "summary.json", result)
    (root / "REPORT.md").write_text(report(result))
    return result


def report(result: dict) -> str:
    """Generate auditable curves; distinguish measured repeats from deterministic simulation."""
    lines = [
        "# P3-T2 队列/背压压测",
        "",
        f"{result['evidence']}。硬件/设备：{result['hardware_device']}。",
        f"每点预热 {result['warmup']} 轮，重复 {result['repeat']} 轮；模拟预热0/重复1。",
        f"env `{result['env_fingerprint']}`；recipe config SHA-256 `{result['config_hash']}`。",
        "成功请求 E2E p50/p95/p99 为 NumPy linear，含 dispatch，排空后统计；超时不填零。",
        "",
        "## 并发–尾时延曲线与背压行为",
        "",
        "| λ/s | C | Q | 成功/到达 | 429 | 超时 | 客户端满 | 窗口 req/s | "
        "实测 p50/p95/p99 ms | 模拟 p99 ms | p99 偏差 | 雪崩轮数/模拟 |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]

    def number(value: float | None) -> str:
        return "null" if value is None else f"{value:.3f}"

    for row in result["rows"]:
        lines.append(
            f"| {row['rate']} | {row['concurrency']} | {row['queue_capacity']} | "
            f"{row['completed']}/{row['arrivals']} | {row['rejected']} | {row['timed_out']} | "
            f"{row['client_full']} | {row['throughput_requests_per_s']:.3f} | "
            + "/".join(number(row["latency_ms"][k]) for k in ("p50", "p95", "p99"))
            + f" | {number(row['simulation']['latency_ms']['p99'])} | "
            + (
                "null"
                if row["p99_relative_deviation"] is None
                else f"{row['p99_relative_deviation']:+.2%}"
            )
            + f" | {row['avalanche_rounds']}/{row['simulation']['avalanche']} |"
        )
    lines += [
        "",
        "## 雪崩点实测 vs 模拟",
        "",
        "T3 OR 规则：p99>1000ms 或末半窗四段时间加权队长均值严格递增、"
        "斜率>1/s 且终点队长≥20；逐轮判定。以下为至少一轮触发的最小扫描 λ；"
        "不是连续临界值，也不以拒绝代替雪崩。",
        "",
        "| C | Q | 实测首触发 λ/s | 模拟首触发 λ/s | λ偏差(模拟−实测) | 首429 λ/s |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    groups = sorted({(r["concurrency"], r["queue_capacity"]) for r in result["rows"]})
    for concurrency, capacity in groups:
        rows = sorted(
            (
                r
                for r in result["rows"]
                if (r["concurrency"], r["queue_capacity"]) == (concurrency, capacity)
            ),
            key=lambda r: r["rate"],
        )
        real = next((r["rate"] for r in rows if r["avalanche_rounds"]), None)
        sim = next((r["rate"] for r in rows if r["simulation"]["avalanche"]), None)
        reject = next((r["rate"] for r in rows if r["rejected"]), None)

        def show(x):
            return "扫描内未观察到" if x is None else str(x)

        lines.append(
            f"| {concurrency} | {capacity} | {show(real)} | {show(sim)} | "
            f"{sim - real if real is not None and sim is not None else '不可估计'} | "
            f"{show(reject)} |"
        )
    lines += [
        "",
        "## 校准对照与客户端开销",
        "",
        "偏差=(模拟p99−实测p99)/实测p99。两侧使用同一请求时刻/长度、seed、λ、C、Q、"
        "窗口和超时；模拟使用 P1-T4 校准的 prefill 斜率和 decode(1)，不是重新拟合 HTTP。",
        "适配器在去掉 C 限制时逐点校验原 T3 的完成/拒绝、分位数与雪崩判定。"
        "有限 C 模型假设 HTTP/429 零耗时；实测有线程、socket、解析和 tracing 扰动。",
        "prefill 固定截距仍省略，decode 仅 context=32 标定；不外推生产服务。",
        "客户端有 C 个预启动线程和至多 C 个 socket，服务 Connection: close，逐请求重连。"
        "满池到达立即丢弃，无隐藏请求队列/重试。低 C 下的低吞吐可能来自客户端限流。",
        "rounds.json 单独保存 dispatch/client thread CPU/transport residual 的分位数、"
        "进程 CPU、队列采样间隔；residual=wire−queue_wait−inference，"
        "包括传输与服务 HTTP 开销，不冒充纯客户端耗时。",
        "",
        "## 产物与配置哈希",
        "",
    ]
    for row in result["rows"]:
        lines.append(f"- `{row['point']}`：config `{row['config_hash']}`。")
    lines += [
        "",
        "每点保存 P0 measurement/env/config/meta、workload、逐轮 requests/queue/summary、"
        "模拟请求/队列/config 和原 T3 summary。根目录保留校准原字节、recipe/env 和汇总。",
        "",
        "## 边界声明",
        "",
        "单进程串行推理、CPU 小模型、loopback、无 TLS。同进程客户端与服务竞争资源；"
        "只测整包响应，流式等价性与 TTFT 见 P3-T1 PR #12。",
        "队长为定时采样近似，短暂变化可能漏采；bounded Q 与有限 C 限制增长，"
        "未触发不证明稳定。时延仅含成功请求，须连同429/超时/客户端满计数解释。",
        "无独立进程重复或置信区间，smoke test 不等于实验证据。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    result = run_load(json.loads(args.config.read_text()), args.runs_dir, check=args.check)
    if args.output:
        args.output.write_text(report(result))
    print(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()

"""模拟: deterministic, single-server discrete-event batching (all times in ms).

FIFO admission; prefill blocks decode and costs prompt_tokens * prefill_ms_per_token.
An iteration budgets prompt tokens plus one decode token per member, including new
members. Prompts are not chunked. Static batches retain their launch width (padded
decode cost) until their last member completes; continuous batches refill only at
iteration boundaries. At a tie: service completion, queue timeout, arrival, batch
timer, observation end, then dispatch. Waiting ends at admission, before prefill.

Arrivals stop at duration_ms; service drains afterwards so latency is uncensored.
Queue averages and primary token throughput cover only the arrival window. Queue
growth is a finite-window warning proxy, never a proof of unbounded growth.
"""

import hashlib
import heapq
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement
from prism.record import save_run


def config_hash(config: dict[str, Any]) -> str:
    """Use the existing measurement protocol's sorted-JSON SHA-256 convention."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> None:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _distribution(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict) or spec.get("kind") not in ("uniform", "truncated_normal"):
        raise ValueError("length distribution must be uniform or truncated_normal")
    for key in ("min", "max"):
        _integer(spec.get(key), key)
    if spec["min"] > spec["max"]:
        raise ValueError("length min exceeds max")
    if spec["kind"] == "truncated_normal":
        mean = spec.get("mean")
        if type(mean) not in (int, float) or not math.isfinite(mean):
            raise ValueError("normal mean must be finite")
        _number(spec.get("std"), "normal std", positive=True)
        normal = NormalDist(mean, spec["std"])
        if spec["min"] != spec["max"] and normal.cdf(spec["min"]) >= normal.cdf(spec["max"]):
            raise ValueError("truncated normal has no numerically resolvable probability mass")


def validate_config(config: dict[str, Any]) -> None:
    """Validate one grid point before generating events or writing records."""
    if not isinstance(config, dict) or config.get("mode") != "模拟":
        raise ValueError("config must explicitly specify mode=模拟")
    if config.get("policy") not in ("static", "continuous"):
        raise ValueError("policy must be static or continuous")
    _integer(config.get("seed"), "seed", 0)
    _integer(config.get("batch_size"), "batch_size")
    _integer(config.get("max_batched_tokens"), "max_batched_tokens")
    for name in ("arrival_rate_per_s", "duration_ms", "p99_threshold_ms"):
        _number(config.get(name), name, positive=True)
    _number(config.get("batch_timeout_ms"), "batch_timeout_ms")
    for name in ("prompt_length", "output_length"):
        _distribution(config.get(name))
    if config["max_batched_tokens"] < max(config["batch_size"], config["prompt_length"]["max"] + 1):
        raise ValueError("token budget must fit the batch width and any prompt plus one decode")
    if config.get("queue_capacity") is not None:
        _integer(config["queue_capacity"], "queue_capacity")
    if config.get("queue_timeout_ms") is not None:
        _number(config["queue_timeout_ms"], "queue_timeout_ms", positive=True)
    model = config.get("service_model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("source"), dict)
        or not model["source"]
    ):
        raise ValueError("service_model requires source provenance")
    _number(model.get("prefill_ms_per_token"), "prefill_ms_per_token")
    nodes = model.get("decode_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("decode_nodes must be nonempty [batch_size, milliseconds] pairs")
    last_batch, last_ms = 0, 0.0
    for node in nodes:
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError("invalid decode node")
        batch, ms = node
        _integer(batch, "decode batch")
        _number(ms, "decode ms", positive=True)
        if batch <= last_batch or ms < last_ms:
            raise ValueError("decode nodes must have increasing batches and nondecreasing times")
        last_batch, last_ms = batch, ms
    if nodes[0][0] != 1 or nodes[-1][0] < config["batch_size"]:
        raise ValueError("decode nodes must cover [1, batch_size]; no endpoint extrapolation")
    growth = config.get("queue_growth")
    if not isinstance(growth, dict):
        raise ValueError("queue_growth requires min_queue and min_slope_per_s")
    _integer(growth.get("min_queue"), "min_queue")
    _number(growth.get("min_slope_per_s"), "min_slope_per_s", positive=True)
    # Reject non-JSON values and non-finite provenance as well.
    config_hash(config)


def sample_length(rng: np.random.Generator, spec: dict[str, Any]) -> int:
    """Uniform integers inclusive; true truncated normal, rounded to nearest integer."""
    low, high = spec["min"], spec["max"]
    if low == high:
        return low
    if spec["kind"] == "uniform":
        return int(rng.integers(low, high + 1))
    normal = NormalDist(spec["mean"], spec["std"])
    left, right = normal.cdf(low), normal.cdf(high)
    probability = float(rng.uniform(left, right))
    probability = min(max(probability, math.nextafter(0.0, 1.0)), math.nextafter(1.0, 0.0))
    return min(high, max(low, round(normal.inv_cdf(probability))))


def decode_ms(model: dict[str, Any], batch_size: int) -> float:
    """Interpolate the configured piecewise-linear decode curve."""
    nodes = model["decode_nodes"]
    if not 1 <= batch_size <= nodes[-1][0]:
        raise ValueError("decode width outside configured nodes")
    return float(np.interp(batch_size, [node[0] for node in nodes], [node[1] for node in nodes]))


@dataclass
class Request:
    id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    status: str = "pending"
    start_ms: float | None = None
    finish_ms: float | None = None
    wait_ms: float | None = None
    latency_ms: float | None = None
    generated_tokens: int = 0


@dataclass
class Simulation:
    """模拟 results contain no wall-clock data; same config gives exact equality."""

    config: dict[str, Any]
    requests: list[Request]
    events: list[dict[str, Any]]
    queue_series: list[dict[str, Any]]
    summary: dict[str, Any]


def _stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return dict.fromkeys(("mean", "stdev", "p50", "p95", "p99"))
    p50, p95, p99 = np.percentile(samples, [50, 95, 99], method="linear")
    return {
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def simulate(config: dict[str, Any]) -> Simulation:
    """Run a finite Poisson arrival window followed by complete FIFO draining."""
    validate_config(config)
    config = json.loads(json.dumps(config, allow_nan=False))
    rng = np.random.default_rng(config["seed"])
    horizon = config["duration_ms"]
    requests = []
    arrival = 0.0
    while True:
        arrival += float(rng.exponential(1000 / config["arrival_rate_per_s"]))
        if arrival >= horizon:
            break
        requests.append(
            Request(
                len(requests),
                arrival,
                sample_length(rng, config["prompt_length"]),
                sample_length(rng, config["output_length"]),
            )
        )

    # Serial numbers make equal-time event ordering explicit and platform stable.
    pending = []
    serial = 0
    priorities = {
        "decode_end": 0,
        "prefill_end": 0,
        "timeout": 1,
        "arrival": 2,
        "batch_timer": 3,
        "horizon": 4,
    }

    def push(time: float, kind: str, request_id: int = -1) -> None:
        nonlocal serial
        heapq.heappush(pending, (time, priorities[kind], serial, kind, request_id))
        serial += 1

    for request in requests:
        push(request.arrival_ms, "arrival", request.id)
    push(horizon, "horizon")
    # dict preserves FIFO order and supports O(1) removal of timed-out requests.
    queue: dict[int, None] = {}
    active: list[int] = []
    events: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    now, previous = 0.0, 0.0
    busy = False
    static_width = 0
    in_system = 0
    max_queue = 0
    queue_area, system_area = 0.0, 0.0
    growth_areas = [0.0] * 4
    horizon_tokens = 0
    timer_head = None

    def emit(kind: str, ids: list[int], **details: Any) -> None:
        nonlocal max_queue
        max_queue = max(max_queue, len(queue))
        events.append({"time_ms": now, "kind": kind, "ids": list(ids), **details})
        series.append({"time_ms": now, "queue_length": len(queue), "in_system": in_system})

    def start_decode() -> None:
        width = static_width if config["policy"] == "static" else len(active)
        cost = decode_ms(config["service_model"], width)
        emit("decode_start", active, width=width, cost_ms=cost)
        push(now + cost, "decode_end")

    emit("start", [])
    while pending:
        now = pending[0][0]
        # Exact piecewise-constant integration, clipped to the arrival window.
        end = min(now, horizon)
        begin = min(previous, horizon)
        queue_area += (end - begin) * len(queue)
        system_area += (end - begin) * in_system
        for index in range(4):
            left = horizon / 2 + index * horizon / 8
            right = left + horizon / 8
            growth_areas[index] += max(0.0, min(end, right) - max(begin, left)) * len(queue)
        previous = now
        # Consume all events at this timestamp before dispatching another iteration.
        while pending and pending[0][0] == now:
            _, _, _, kind, rid = heapq.heappop(pending)
            if kind == "arrival":
                request = requests[rid]
                capacity = config.get("queue_capacity")
                if capacity is not None and len(queue) >= capacity:
                    request.status = "rejected"
                    request.finish_ms = now
                    emit("reject", [rid])
                else:
                    request.status = "queued"
                    queue[rid] = None
                    in_system += 1
                    emit("arrival", [rid])
                    if config.get("queue_timeout_ms") is not None:
                        push(now + config["queue_timeout_ms"], "timeout", rid)
            elif kind == "timeout":
                if rid not in queue:
                    continue
                del queue[rid]
                request = requests[rid]
                request.status = "timed_out"
                request.finish_ms = now
                request.wait_ms = now - request.arrival_ms
                in_system -= 1
                emit("timeout", [rid])
            elif kind == "prefill_end":
                emit("prefill_end", active)
                start_decode()
            elif kind == "decode_end":
                finished = []
                for rid in active:
                    request = requests[rid]
                    request.generated_tokens += 1
                    if now <= horizon:
                        horizon_tokens += 1
                    if request.generated_tokens == request.output_tokens:
                        request.status = "completed"
                        request.finish_ms = now
                        request.latency_ms = now - request.arrival_ms
                        in_system -= 1
                        finished.append(rid)
                emit("decode_end", active, completed=finished)
                active = [rid for rid in active if requests[rid].status == "active"]
                busy = False
            elif kind == "batch_timer":
                if not queue or rid != next(iter(queue)):
                    continue
                timer_head = None
                emit("batch_timer", [rid])
            else:
                emit("horizon", [])

        if busy:
            continue
        if config["policy"] == "static" and not active and queue:
            head = next(iter(queue))
            deadline = requests[head].arrival_ms + config["batch_timeout_ms"]
            if len(queue) < config["batch_size"] and now < deadline:
                if timer_head != head:
                    push(deadline, "batch_timer", head)
                    timer_head = head
                continue

        admitted = []
        budget = len(active)
        # A static batch never refills before its last member completes.
        if config["policy"] == "continuous" or not active:
            while queue and len(active) < config["batch_size"]:
                rid = next(iter(queue))
                request = requests[rid]
                tokens = request.prompt_tokens + 1
                if budget + tokens > config["max_batched_tokens"]:
                    break
                del queue[rid]
                budget += tokens
                request.status = "active"
                request.start_ms = now
                request.wait_ms = now - request.arrival_ms
                active.append(rid)
                admitted.append(rid)
        if not active:
            continue
        if config["policy"] == "static" and admitted:
            static_width = len(active)
        busy = True
        prefill_tokens = sum(requests[rid].prompt_tokens for rid in admitted)
        prefill_cost = prefill_tokens * config["service_model"]["prefill_ms_per_token"]
        emit(
            "iteration",
            active,
            admitted=admitted,
            batched_tokens=budget,
            prefill_tokens=prefill_tokens,
            prefill_ms=prefill_cost,
        )
        if prefill_cost:
            push(now + prefill_cost, "prefill_end")
        else:
            start_decode()

    completed = [request for request in requests if request.status == "completed"]
    latency = _stats([request.latency_ms for request in completed])
    waiting = _stats([request.wait_ms for request in completed])
    end_ms = max([horizon, *(request.finish_ms for request in requests)])
    growth_means = [area / (horizon / 8) for area in growth_areas]
    slope = (growth_means[-1] - growth_means[0]) / (3 * horizon / 8 / 1000)
    horizon_queue = next(
        item["queue_length"] for item in reversed(series) if item["time_ms"] <= horizon
    )
    growth = config["queue_growth"]
    growing = (
        all(right > left for left, right in zip(growth_means[:-1], growth_means[1:], strict=True))
        and slope > growth["min_slope_per_s"]
        and horizon_queue >= growth["min_queue"]
    )
    reasons = []
    if latency["p99"] is not None and latency["p99"] > config["p99_threshold_ms"]:
        reasons.append("p99_threshold")
    if growing:
        reasons.append("finite_window_queue_growth_proxy")
    summary = {
        "mode": "模拟",
        "config_hash": config_hash(config),
        "arrivals": len(requests),
        "completed": len(completed),
        "rejected": sum(request.status == "rejected" for request in requests),
        "timed_out": sum(request.status == "timed_out" for request in requests),
        "wait_ms": waiting,
        "latency_ms": latency,
        "max_queue_length": max_queue,
        "mean_queue_length": queue_area / horizon,
        "mean_in_system": system_area / horizon,
        "window_output_tokens": horizon_tokens,
        "throughput_tokens_per_s": horizon_tokens / (horizon / 1000),
        "completed_output_tokens": sum(request.output_tokens for request in completed),
        "drained_throughput_tokens_per_s": sum(request.output_tokens for request in completed)
        / (end_ms / 1000),
        "observation_ms": horizon,
        "drained_until_ms": end_ms,
        "late_window_mean_queues": growth_means,
        "queue_slope_per_s": slope,
        "avalanche": bool(reasons),
        "avalanche_reasons": reasons,
    }
    return Simulation(config, requests, events, series, summary)


def save_simulation(result: Simulation, runs_dir: str | Path, slug: str) -> Path:
    """Persist 模拟 E2E statistics through save_run, plus full simulation artifacts.

    Measurement is a compatibility envelope, not a call to measure(): timings are
    simulated request latencies, memory fields are zero (not measured). Its fixed
    schema retains E2E samples and quantiles; summary.json holds the other metrics.
    Empty completion sets cannot form a valid Measurement and are rejected before
    writing, rather than inventing a latency sample.
    """
    samples = [request.latency_ms for request in result.requests if request.status == "completed"]
    if not samples:
        raise ValueError("模拟: cannot save Measurement without completed requests")
    measurement = Measurement(
        samples_ms=samples,
        **result.summary["latency_ms"],
        peak_tracemalloc_bytes=0,
        max_rss_bytes=0,
        config_hash=config_hash(result.config),
        env_fingerprint=env_fingerprint(collect_env()),
        captured_at=datetime.now(UTC).isoformat(),
    )
    path = save_run(runs_dir, slug, result.config, measurement)
    artifacts = {
        "summary.json": result.summary,
        "requests.json": {"mode": "模拟", "requests": [asdict(r) for r in result.requests]},
        "events.json": {"mode": "模拟", "events": result.events},
        "queue.json": {"mode": "模拟", "queue_series": result.queue_series},
    }
    for name, content in artifacts.items():
        (path / name).write_text(
            json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return path

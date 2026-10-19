"""Measure synchronous callables with a reproducible timing and memory protocol."""

import hashlib
import json
import resource
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np

from prism.env import collect_env, env_fingerprint


@dataclass
class Measurement:
    """Retain timing samples, summary statistics, memory peaks, and provenance.

    Timing values are milliseconds. Quantiles use NumPy's linear interpolation;
    stdev is the sample standard deviation (ddof=1), or 0.0 for one sample.
    max_rss_bytes is the process lifetime high-water mark in bytes on macOS,
    provided only as a reference; its semantics on other platforms are undefined.
    """

    samples_ms: list[float]
    mean: float
    stdev: float
    p50: float
    p95: float
    p99: float
    peak_tracemalloc_bytes: int
    max_rss_bytes: int
    config_hash: str
    env_fingerprint: str
    captured_at: str

    def to_json(self) -> str:
        """Serialize all fields as sorted JSON with two-space indentation."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, value: str) -> Self:
        """Restore a measurement serialized by to_json."""
        return cls(**json.loads(value))


def measure[T](
    fn: Callable[[], T], *, warmup: int, repeat: int, config: dict[str, Any]
) -> Measurement:
    """Warm up fn, then retain repeat perf_counter_ns timings in milliseconds.

    warmup and repeat must be non-bool integers, with warmup >= 0 and repeat >= 1.
    The sorted JSON config hash is computed before warmup; NaN and Inf are rejected.
    Tracing must be inactive on entry: this function exclusively owns tracemalloc
    during the timed region and stops it even if fn raises. Warmup is untraced.
    Tracing and sample bookkeeping affect timings and the traced peak; T4 will
    calibrate this overhead. Tracemalloc does not cover all native allocations.
    Quantiles use np.percentile(method="linear"); stdev uses ddof=1 (one sample: 0.0).
    max_rss_bytes is read after timing and is a process lifetime high-water mark,
    in bytes on macOS, for reference only; other platforms are undefined.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-bool integer >= 0")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("repeat must be a non-bool integer >= 1")
    if tracemalloc.is_tracing():
        raise RuntimeError("measure requires exclusive ownership of tracemalloc")

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            fn()
            end_ns = time.perf_counter_ns()
            samples_ms.append((end_ns - start_ns) / 1_000_000)
        _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    p50, p95, p99 = np.percentile(samples_ms, [50, 95, 99], method="linear")
    return Measurement(
        samples_ms=samples_ms,
        mean=statistics.mean(samples_ms),
        stdev=statistics.stdev(samples_ms) if repeat > 1 else 0.0,
        p50=float(p50),
        p95=float(p95),
        p99=float(p99),
        peak_tracemalloc_bytes=peak_tracemalloc_bytes,
        max_rss_bytes=max_rss_bytes,
        config_hash=config_hash,
        env_fingerprint=env_fingerprint(collect_env()),
        captured_at=datetime.now(UTC).isoformat(),
    )

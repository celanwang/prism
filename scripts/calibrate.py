"""Calibrate T3's timed-sample perturbation; never benchmark a business SUT."""

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement, measure

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "calibrate.json"
REPORT_PATH = ROOT / "results" / "p0" / "calibration.md"
STATISTICS = ("mean", "p50", "p95", "p99")


@dataclass
class Timing:
    """Bare timings in milliseconds, using T3's linear quantile convention."""

    samples_ms: list[float]
    mean: float
    p50: float
    p95: float
    p99: float


@dataclass
class Comparison:
    name: str
    bare: Timing
    traced: Measurement

    def differences(self) -> dict[str, float]:
        return {key: getattr(self.traced, key) - getattr(self.bare, key) for key in STATISTICS}

    def threshold_ms(self) -> float:
        """Conservative empirical floor, not a statistical detection guarantee."""
        return max(
            *(abs(value) for value in self.differences().values()),
            self.bare.p99 - self.bare.p50,
            self.traced.p99 - self.traced.p50,
        )


def validate_config(config: dict[str, Any]) -> None:
    """Reject malformed recipes before running either timing arm."""
    if not isinstance(config, dict) or set(config) != {"warmup", "repeat", "probes"}:
        raise ValueError("config requires exactly warmup, repeat, probes")

    def integer(value: Any, minimum: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    integer(config["warmup"], 0, "warmup")
    integer(config["repeat"], 1, "repeat")
    probes = config["probes"]
    expected = {
        "noop": set(),
        "compute": {"elements", "initial", "increment"},
        "allocation": {"objects", "bytes_per_object"},
    }
    if not isinstance(probes, dict) or set(probes) != set(expected):
        raise ValueError("probes requires noop, compute, allocation")
    for name, keys in expected.items():
        if not isinstance(probes[name], dict) or set(probes[name]) != keys:
            raise ValueError(f"invalid parameters for {name}")
    integer(probes["compute"]["elements"], 1, "elements")
    for key in ("initial", "increment"):
        value = probes["compute"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite numeric")
    for key in ("objects", "bytes_per_object"):
        integer(probes["allocation"][key], 1, key)


def make_probe(name: str, parameters: dict[str, Any]) -> Callable[[], Any]:
    """Create each arm separately so mutable compute state starts identically.

    Compute uses one native arithmetic operation on a preallocated float64 array:
    no per-call payload/output buffers or Python scalar arithmetic allocations.
    Python/NumPy call machinery can still have small internal allocations.
    Allocation returns a fresh list of distinct bytearrays, freed by the caller.
    """
    if name == "noop":
        return lambda: None
    if name == "compute":
        values = np.full(parameters["elements"], parameters["initial"], dtype=np.float64)
        increment = np.float64(parameters["increment"])

        def compute() -> np.ndarray:
            return np.add(values, increment, out=values)

        return compute
    if name == "allocation":
        objects = parameters["objects"]
        size = parameters["bytes_per_object"]

        def allocate() -> list[bytearray]:
            return [bytearray(size) for _ in range(objects)]

        return allocate
    raise ValueError(f"unknown probe: {name}")


def bare_timing(fn: Callable[[], Any], *, warmup: int, repeat: int) -> Timing:
    """Time directly without measure() or tracing, including result disposal."""
    if tracemalloc.is_tracing():
        raise RuntimeError("bare timing requires tracemalloc to be inactive")
    for _ in range(warmup):
        fn()
    samples_ms = []
    for _ in range(repeat):
        start_ns = time.perf_counter_ns()
        fn()
        end_ns = time.perf_counter_ns()
        samples_ms.append((end_ns - start_ns) / 1_000_000)
    p50, p95, p99 = np.percentile(samples_ms, [50, 95, 99], method="linear")
    return Timing(samples_ms, statistics.mean(samples_ms), float(p50), float(p95), float(p99))


def calibrate(config: dict[str, Any]) -> list[Comparison]:
    """Run one bare arm then one traced arm per configured probe."""
    validate_config(config)
    results = []
    # Match T3's order-independent config hash with an order-independent recipe.
    for name in ("noop", "compute", "allocation"):
        parameters = config["probes"][name]
        bare = bare_timing(
            make_probe(name, parameters), warmup=config["warmup"], repeat=config["repeat"]
        )
        traced = measure(
            make_probe(name, parameters),
            warmup=config["warmup"],
            repeat=config["repeat"],
            config=config,
        )
        results.append(Comparison(name, bare, traced))
    return results


def cpu_model() -> str:
    """Supplement T1's unchanged environment fingerprint with a CPU model."""
    if platform.system() == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return platform.processor() or platform.machine()


def render_report(
    config: dict[str, Any], results: list[Comparison], env: dict[str, Any], hardware: str
) -> str:
    """Render observed differences and an explicitly heuristic resolution floor."""
    fingerprint = env_fingerprint(env)
    hashes = {result.traced.config_hash for result in results}
    if len(hashes) != 1 or any(result.traced.env_fingerprint != fingerprint for result in results):
        raise RuntimeError("calibration provenance changed between arms")
    lines = [
        "# P0-T4 测量器自身开销标定",
        "",
        f"- env 指纹：`{fingerprint}`",
        f"- config 哈希（T3 sorted JSON SHA-256）：`{next(iter(hashes))}`",
        f"- 采集时间（UTC）：{env['captured_at']}",
        f"- 硬件/设备：{hardware}；CPU / {env['platform']['machine']}；",
        f"  逻辑 CPU 数 {env['cpu_count']}；物理内存 {env['memory_bytes']} bytes。",
        f"- 每个探针、每个 arm：预热 {config['warmup']} 次，重复 {config['repeat']} 次；",
        "  共一轮，顺序为裸计时 → measure()；CPU 实测，未使用 MPS/CUDA。",
        "- 所有时延单位为 ms；p50/p95/p99 使用 NumPy linear 插值。",
        "",
        "## 方法与边界",
        "",
        "裸计时直接调用 perf_counter_ns，不经 measure()，且确认 tracing 未启动。",
        "两侧分别构造等价探针，使用相同预热、重复次数、结果释放和样本簿记方式；",
        "measure() 沿用 T3，在计时区间启用 tracemalloc。差值为 measure − bare，保留负值。",
        "本标定比较函数调用区间的扰动；不含 measure() 的启动/停止 tracing、环境采集、",
        "统计汇总等整次 API 墙钟成本。样本簿记位于单次计时外，但可间接扰动后续样本。",
        "",
        "noop 为 lambda: None；compute 为预分配 float64 缓冲区上的原位 NumPy 加法闭包，",
        "无每次调用的数据/输出缓冲区分配（Python/NumPy 调用层仍可能有少量内部开销）。",
        "allocation 每次新建配置数量的 bytearray 小对象及一个列表，并在计时内释放；",
        "因此其差值包含分配和释放的追踪成本，不能解释为单个纯分配操作的成本。",
        "",
        "## 对照与开销",
        "",
        "| 探针 | 统计量 | 裸计时 ms | measure() ms | 开销 Δ ms | 放大倍数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        for key, difference in result.differences().items():
            bare = getattr(result.bare, key)
            traced = getattr(result.traced, key)
            factor = f"{traced / bare:.3f}×" if bare else "n/a（裸计时为零）"
            lines.append(
                f"| {result.name} | {key} | {bare:.9f} | {traced:.9f} | "
                f"{difference:+.9f} | {factor} |"
            )
    lines.extend(["", "## 可分辨最小收益", ""])
    for result in results:
        lines.append(f"- {result.name} 的经验分辨率阈值：{result.threshold_ms():.9f} ms。")
    threshold = max(result.threshold_ms() for result in results)
    # Round upward, retaining a positive safety floor even for all-zero mock clocks.
    suggested = max(1, math.ceil(threshold * 1_000_000)) / 1_000_000
    lines.extend(
        [
            "",
            "每个探针的阈值取 mean/p50/p95/p99 差值绝对值，以及两侧 p99−p50 尾部跨度的最大值。",
            "全局建议取三者最大值并向上取整到 0.000001 ms；这是保守经验噪声/扰动尺度，",
            "不是通过显著性检验得到的统计最小可检测效应，也不是计时器的硬件分辨率。",
            f"在本环境、本配方覆盖的负载范围，可分辨最小收益按约 {suggested:.6f} ms 量级处理；",
            f"建议：**低于 {suggested:.6f} ms 的差异不作为证据**。",
            "超过该阈值也不自动证明收益，应在匹配分配行为的负载上独立重复标定和实验。",
            "本轮未随机化顺序、未估计置信区间，温度、频率与系统调度漂移仍可影响差值；",
            "负差值是本轮观测，不表示 tracing 必然加速。单一分配量的放大倍数不能外推为",
            "随分配量增长的函数；这些数字仅描述测量器，不构成任何 SUT 性能结论。",
            "",
            "## 配方与环境锚点",
            "",
            "复现：`uv run --locked python scripts/calibrate.py`（覆盖本报告）；",
            "`uv run --locked python scripts/calibrate.py --check` 仅打印本次实测报告，不写入。",
            "config 哈希直接取自 measure()；env 指纹复用 T1，CPU 型号是额外描述字段。",
            "",
            "```json",
            json.dumps(config, sort_keys=True, indent=2),
            "```",
            "",
            "```json",
            json.dumps(env, sort_keys=True, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check", action="store_true", help="print report without writing files")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = calibrate(config)
    report = render_report(config, results, collect_env(), cpu_model())
    if args.check:
        print(report, end="")
    else:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(REPORT_PATH)


if __name__ == "__main__":
    main()

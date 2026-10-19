"""Measure the P0 SUT, persist every workload, and render only validated runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline.json"
REPORT_PATH = ROOT / "results" / "p0" / "REPORT.md"
RUNS_PATH = ROOT / "runs"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass
class Workload:
    name: str
    kind: str
    parameters: dict[str, Any]
    fn: Callable[[], Any]
    tokens: int
    warmup: int
    repeat: int


def integer(value: Any, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported or incomplete recipes before measurement or writing."""
    keys = {"task", "phase", "sut_path", "device", "threads", "calibration", "prefill", "decode"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f"recipe requires exactly {sorted(keys)}")
    if config["task"] != "P0-T6" or config["phase"] != "P0" or config["device"] != "cpu":
        raise ValueError("only P0-T6 / P0 / cpu is supported")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("baseline requires one thread")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("sut_path must be a nonempty path")
    expected = {
        "calibration": {"report", "resolution_ms", "rationale"},
        "prefill": {"sequence_lengths", "token_id", "warmup", "repeat"},
        "decode": {"prompt_ids", "max_new_tokens", "warmup", "repeat"},
    }
    for name, fields in expected.items():
        if not isinstance(config[name], dict) or set(config[name]) != fields:
            raise ValueError(f"invalid {name} fields")
    calibration = config["calibration"]
    threshold = calibration["resolution_ms"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("resolution_ms must be finite and positive")
    for key in ("report", "rationale"):
        if not isinstance(calibration[key], str) or not calibration[key]:
            raise ValueError(f"calibration.{key} must be nonempty text")
    for kind in ("prefill", "decode"):
        integer(config[kind]["warmup"], 0, f"{kind}.warmup")
        integer(config[kind]["repeat"], 1, f"{kind}.repeat")
    lengths = config["prefill"]["sequence_lengths"]
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("sequence_lengths must be a nonempty list")
    for length in lengths:
        integer(length, 1, "sequence length")
    if len(set(lengths)) != len(lengths):
        raise ValueError("sequence_lengths must be unique")
    integer(config["prefill"]["token_id"], 0, "token_id")
    prompt = config["decode"]["prompt_ids"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt_ids must be nonempty")
    for token in prompt:
        integer(token, 0, "prompt token")
    integer(config["decode"]["max_new_tokens"], 1, "max_new_tokens")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def configure_threads(config: dict[str, Any]) -> None:
    """Set native thread limits before importing NumPy in the CLI process."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config["threads"])


def make_workloads(sut: SUT, config: dict[str, Any]) -> list[Workload]:
    """Build fixed batch-one inputs outside timing; decode always disables EOS stop."""
    import numpy as np

    validate_config(config)
    prefill, decode = config["prefill"], config["decode"]
    if max(prefill["sequence_lengths"]) > sut.config.context_length:
        raise ValueError("prefill exceeds SUT context_length")
    if max(prefill["token_id"], *decode["prompt_ids"]) >= sut.config.vocab_size:
        raise ValueError("recipe token outside SUT vocabulary")
    workloads = []
    for length in prefill["sequence_lengths"]:
        tokens = np.full((1, length), prefill["token_id"], dtype=np.int64)
        workloads.append(
            Workload(
                f"prefill-t{length}",
                "prefill",
                {"sequence_length": length, "token_id": prefill["token_id"], "batch_size": 1},
                lambda tokens=tokens: sut.forward(tokens),
                length,
                prefill["warmup"],
                prefill["repeat"],
            )
        )
    prompt = list(decode["prompt_ids"])
    workloads.append(
        Workload(
            "decode",
            "decode",
            {"prompt_ids": prompt, "max_new_tokens": decode["max_new_tokens"], "eos_id": None},
            lambda: sut.generate(prompt, max_new_tokens=decode["max_new_tokens"], eos_id=None),
            decode["max_new_tokens"],
            decode["warmup"],
            decode["repeat"],
        )
    )
    return workloads


def cpu_model() -> str:
    if platform.system() == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return platform.processor() or platform.machine()


def run_benchmark(config: dict[str, Any], runs_dir: Path) -> list[Path]:
    """Use the existing measurement and recording APIs without altering protocol."""
    validate_config(config)
    configure_threads(config)
    # Keep NumPy/SUT imports after thread configuration for direct CLI execution.
    from prism.measure import measure
    from prism.record import save_run
    from prism.sut import load_sut

    sut_path = ROOT / config["sut_path"]
    sut = load_sut(sut_path)
    workloads = make_workloads(sut, config)
    sut_info = {
        "config": asdict(sut.config),
        "parameter_count": sum(value.size for value in sut.arrays.values()),
        "manifest_sha256": hashlib.sha256((sut_path / "MANIFEST.json").read_bytes()).hexdigest(),
    }
    runtime = {
        "cpu_model": cpu_model(),
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
    }
    batch = datetime.now(UTC).strftime("%H%M%S%f")
    paths = []
    for workload in workloads:
        record_config = {
            "task": config["task"],
            "phase": config["phase"],
            "recipe": config,
            "sut": sut_info,
            "runtime": runtime,
            "workload": {
                "name": workload.name,
                "kind": workload.kind,
                "parameters": workload.parameters,
                "tokens": workload.tokens,
                "warmup": workload.warmup,
                "repeat": workload.repeat,
            },
        }
        measurement = measure(
            workload.fn, warmup=workload.warmup, repeat=workload.repeat, config=record_config
        )
        paths.append(
            save_run(runs_dir, f"baseline-{batch}-{workload.name}", record_config, measurement)
        )
    return paths


def matching_runs(runs_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Choose the latest complete saved batch for this exact recipe; never mix batches."""
    from prism.record import load_run

    expected = {f"prefill-t{length}" for length in config["prefill"]["sequence_lengths"]} | {
        "decode"
    }
    batches: dict[str, dict[str, Path]] = {}
    for path in sorted(runs_dir.glob("*-baseline-*")):
        run = load_run(path)
        if run.config.get("recipe") != config:
            continue
        batch = path.name.rsplit("-" + run.config["workload"]["name"], 1)[0]
        batches.setdefault(batch, {})[run.config["workload"]["name"]] = path
    for batch in sorted(batches, reverse=True):
        if set(batches[batch]) == expected:
            return list(batches[batch].values())
    raise ValueError("no complete saved baseline matches recipe; run without --check first")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def render_report(paths: list[Path]) -> str:
    """Load all report values from saved records, including derived throughput/deltas."""
    from prism.record import load_run

    runs = [load_run(path) for path in paths]
    if not runs:
        raise ValueError("report requires saved runs")
    first = runs[0]
    recipe = first.config["recipe"]
    validate_config(recipe)
    expected = [f"prefill-t{length}" for length in recipe["prefill"]["sequence_lengths"]] + [
        "decode"
    ]
    by_name = {run.config["workload"]["name"]: run for run in runs}
    if len(by_name) != len(runs) or set(by_name) != set(expected):
        raise ValueError("report requires exactly one record per configured workload")
    batches = {
        (
            run.path.parent.resolve(),
            run.path.name.rsplit("-" + run.config["workload"]["name"], 1)[0],
        )
        for run in runs
    }
    if len(batches) != 1:
        raise ValueError("baseline provenance requires a single measurement batch")
    for run in runs:
        if (
            run.config["recipe"] != recipe
            or run.config["sut"] != first.config["sut"]
            or run.config["runtime"] != first.config["runtime"]
            or run.measurement.env_fingerprint != first.measurement.env_fingerprint
        ):
            raise ValueError("baseline provenance differs between workloads")
        workload = run.config["workload"]
        if len(run.measurement.samples_ms) != workload["repeat"]:
            raise ValueError("sample count differs from recorded repeat")
    runs = [by_name[name] for name in expected]
    env = first.env
    sut_info = first.config["sut"]
    threshold = recipe["calibration"]["resolution_ms"]
    lines = [
        "# P0 首份基准报告",
        "",
        "## P0 阶段验收总览",
        "",
        "| 任务 | 交付物与验收证据指针 |",
        "| --- | --- |",
        "| T1 环境锁定 | `src/prism/env.py`、`configs/env.lock`、`tests/test_env.py`；"
        "本地合并提交 `86fd087`（无 PR 号） |",
        "| T2 SUT | `src/prism/sut.py`、`data/sut/kaleido-p3/MANIFEST.json`、"
        "`tests/test_sut.py`；[PR #2](https://github.com/celanwang/prism/pull/2) |",
        "| T3 测量协议 | `src/prism/measure.py`、`tests/test_measure.py`；"
        "[PR #1](https://github.com/celanwang/prism/pull/1) |",
        "| T4 标定 | [标定报告](calibration.md)、`configs/calibrate.json`、"
        "`tests/test_calibrate.py`；[PR #4](https://github.com/celanwang/prism/pull/4) |",
        "| T5 记录 | `src/prism/record.py`、`tests/test_record.py`；"
        "[PR #3](https://github.com/celanwang/prism/pull/3) |",
        "| T6 基准闭环 | `scripts/baseline.py`、`configs/baseline.json`、"
        "`tests/test_baseline.py`、本报告及下列 runs；当前分支交付，尚无 PR |",
        "",
        "T1–T5 已合并；T6 提供实测证据和复现入口，P0 最终验收由规划层独立重跑后决定。",
        "",
        "## 环境与测量配方",
        "",
        f"- 硬件/设备：{first.config['runtime']['cpu_model']}；CPU / "
        f"{env['platform']['machine']}；{recipe['threads']} 线程；"
        f"逻辑 CPU 数 {env['cpu_count']}；物理内存 {env['memory_bytes']} bytes。",
        f"- env 指纹：`{first.measurement.env_fingerprint}`。",
        f"- SUT：{sut_info['parameter_count']} 参数（约 "
        f"{sut_info['parameter_count'] / 1_000_000:.2f}M）；float32；"
        f"结构 `{json.dumps(sut_info['config'], sort_keys=True)}`。",
        f"- SUT manifest SHA-256：`{sut_info['manifest_sha256']}`。",
        "- 单进程顺序执行；CLI 在导入 NumPy 前设置各 BLAS/OpenMP 线程上限："
        f"`{json.dumps(first.config['runtime']['thread_environment'], sort_keys=True)}`。",
        "- 时延单位 ms；p50/p95/p99 由 measure() 使用 NumPy linear 插值；"
        "以下为单轮样本分位数，不是置信区间。",
        f"- 预热/重复依据：{recipe['calibration']['rationale']}",
        "",
        "## 基准数字表",
        "",
        "吞吐为推算值：prefill 使用序列 token 数 × 1000 / p50(ms)；"
        "decode 使用新增 token 数 × 1000 / 端到端 p50(ms)，不含 prompt token 数。"
        "这是 p50 时延倒数推算，非平均吞吐，也不是逐 token 时延分位数。",
        "",
        "| 负载 | warmup | repeat | p50 ms | p95 ms | p99 ms | 推算 token/s | "
        "peak_tracemalloc_bytes | max_rss_bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        m, w = run.measurement, run.config["workload"]
        throughput = f"{w['tokens'] * 1000 / m.p50:.6f}" if m.p50 else "n/a"
        lines.append(
            f"| {w['name']} | {w['warmup']} | {w['repeat']} | {m.p50:.9f} | {m.p95:.9f} | "
            f"{m.p99:.9f} | {throughput} | {m.peak_tracemalloc_bytes} | {m.max_rss_bytes} |"
        )
    lines.extend(
        [
            "",
            "内存与时延来自同一次 measure()，没有独立计时器。peak_tracemalloc_bytes 是"
            "重复测量区间的 traced 峰值，含输出临时分配与样本簿记，不覆盖全部原生分配；"
            "模型和固定输入在 tracing 前加载。max_rss_bytes 是 macOS 进程生命周期高水位，"
            "包含模型加载及之前负载，不能解释为各负载独立增量或相加。",
            "",
            "### 记录与配置哈希",
            "",
            "config 哈希来自 measure() 的完整 config（sorted JSON SHA-256），包括配方、"
            "负载、SUT 与线程设置；记录路径和采集时间不参与 config 哈希。",
            "每目录包含 config.json、measurement.json（全部样本）、env.lock、meta.json。",
            "",
            "| 负载 | config SHA-256 | 采集时间 UTC | runs 目录 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        lines.append(
            f"| {run.config['workload']['name']} | `{run.measurement.config_hash}` | "
            f"{run.measurement.captured_at} | `{display_path(run.path)}` |"
        )
    lines.extend(
        [
            "",
            "## T4 分辨率对照",
            "",
            f"沿用 `{recipe['calibration']['report']}` 的经验阈值 {threshold:.6f} ms。"
            "它是噪声/扰动尺度，超过阈值仍不能自动证明收益；增加重复次数不能消除 tracing 扰动。",
            "",
            "| 负载 | p50 / 阈值 | p95−p50 ms | 判定 | p99−p50 ms | 判定 |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )

    def verdict(delta: float) -> str:
        return "不可分辨" if abs(delta) < threshold else "超过经验阈值，非显著性证明"

    for run in runs:
        m = run.measurement
        d95, d99 = m.p95 - m.p50, m.p99 - m.p50
        lines.append(
            f"| {run.config['workload']['name']} | {m.p50 / threshold:.6f} | "
            f"{d95:.9f} | {verdict(d95)} | {d99:.9f} | {verdict(d99)} |"
        )
    lines.extend(
        [
            "",
            "| prefill p50 对照 | 绝对差 ms | 判定 |",
            "| --- | ---: | --- |",
        ]
    )
    for left, right in combinations(runs[:-1], 2):
        delta = abs(left.measurement.p50 - right.measurement.p50)
        lines.append(
            f"| {left.config['workload']['name']} / {right.config['workload']['name']} | "
            f"{delta:.9f} | {verdict(delta)} |"
        )
    lines.extend(
        [
            "",
            "不同序列长度是不同工作量，上表只作分辨率对照，不是优化收益对照。"
            "低于经验阈值的差异标为“不可分辨”，不得作为性能收益证据。",
            "",
            "## 边界声明",
            "",
            "本轮为 CPU/单线程实测，未使用 MPS/CUDA。SUT 无 KV cache，"
            "注意力采用 O(n²) 朴素前向；decode 每一步重算完整上下文，超过窗口后按 SUT 规则裁剪。"
            "decode 禁用 EOS 提前停止，固定生成配方指定的新 token 数；计时包含 prompt 列表复制、"
            "逐步窗口构造、前向与 argmax，不含模型加载、tokenizer 编码和文本解码。"
            "prefill 是 batch-one 单次前向，固定输入构造在计时外。"
            "这些数字是协议正确性证据，不是生产级性能结论；tracemalloc 扰动、"
            "系统调度与温度漂移仍影响观测，单轮结果不支持普遍性能结论。",
            "",
            "## 复现与验收入口",
            "",
            "- `uv run --locked python scripts/baseline.py`：测量配方、保存新批次记录、覆盖报告。",
            "- `uv run --locked python scripts/baseline.py --check`：读取最新匹配的完整批次并打印，"
            "不测量、不写文件；已保存的记录是报告的唯一数字来源。",
            "- `uv run --locked python scripts/baseline.py --from-runs <目录> ...`："
            "从指定完整批次幂等重建报告；加 `--check` 可仅打印。",
            "- 验收：`uv sync --locked`、`uv run --locked pytest`、"
            "`uv run --locked ruff check .`、`uv run --locked ruff format --check .`、"
            "上述 `--check` 与 `git status --short`。测试中的极小配方仅为 smoke，"
            "本报告数字来自正式配方记录。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--from-runs", type=Path, nargs="+")
    parser.add_argument("--check", action="store_true", help="print saved evidence without writing")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.from_runs:
        from prism.record import load_run

        paths = args.from_runs
        if any(load_run(path).config.get("recipe") != config for path in paths):
            raise ValueError("saved recipe does not match --config")
    elif args.check:
        paths = matching_runs(args.runs_dir, config)
    else:
        paths = run_benchmark(config, args.runs_dir)
    report = render_report(paths)
    if args.check:
        print(report, end="")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(args.report)


if __name__ == "__main__":
    main()

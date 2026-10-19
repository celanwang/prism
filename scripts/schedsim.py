"""Run the P1-T3 λ × B grid; all reported values are 模拟, not device measurements."""

import argparse
import json
import platform
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prism.schedsim import config_hash, save_simulation, simulate, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p1_schedsim.json"


def grid_points(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand and validate the complete grid before any simulation or write."""
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a JSON object")
    axes = ("arrival_rates_per_s", "batch_sizes", "policies")
    for key in axes:
        values = recipe.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} must contain unique values")
    points = []
    for rate in recipe["arrival_rates_per_s"]:
        for batch in recipe["batch_sizes"]:
            for policy in recipe["policies"]:
                point = {key: deepcopy(value) for key, value in recipe.items() if key not in axes}
                point.update(arrival_rate_per_s=rate, batch_size=batch, policy=policy)
                validate_config(point)
                points.append(point)
    return points


def run_grid(recipe: dict[str, Any], runs_dir: Path, *, check: bool = False) -> str:
    """--check caps the workload, compares every event twice, and never saves."""
    recipe = deepcopy(recipe)
    if check:
        recipe["duration_ms"] = min(recipe["duration_ms"], 500)
        recipe["arrival_rates_per_s"] = recipe["arrival_rates_per_s"][:2]
        recipe["batch_sizes"] = recipe["batch_sizes"][:2]
    points = grid_points(recipe)
    lines = [
        "模拟 — continuous batching 调度扫描" + ("（小规模自检；非实验证据）" if check else ""),
        f"模拟宿主: {platform.system()} / {platform.machine()}; 单进程串行；"
        "设备为解析服务模型，无真实推理；预热 0，重复 1。",
        f"扫描配方 SHA-256: {config_hash(recipe)}",
        "p50/p95/p99: 完成请求 E2E ms（含排空）；窗口 token/s: 到达窗口内生成 token。",
        "雪崩: p99 阈值或有限窗口队长增长代理；参数来源与限制见各记录 config.json。",
        "measurement.json 为模拟 E2E 统计；summary/requests/events/queue.json 为完整附加产物。",
        "",
        "| λ /s | B | 策略 | p50 ms | p95 ms | p99 ms | 窗口 token/s | "
        "最大队长 | 拒绝 | 超时 | 雪崩 | config SHA-256 | runs |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    batch_id = datetime.now(UTC).strftime("%H%M%S%f")
    for index, point in enumerate(points):
        result = simulate(point)
        if check and result != simulate(point):
            raise AssertionError("模拟: same-seed event determinism check failed")
        path = "不写入"
        if not check:
            path = str(save_simulation(result, runs_dir, f"schedsim-{batch_id}-{index}"))
        summary = result.summary
        quantiles = [summary["latency_ms"][key] for key in ("p50", "p95", "p99")]
        values = " | ".join("n/a" if value is None else f"{value:.6f}" for value in quantiles)
        lines.append(
            f"| {point['arrival_rate_per_s']} | {point['batch_size']} | {point['policy']} | "
            f"{values} | {summary['throughput_tokens_per_s']:.6f} | "
            f"{summary['max_queue_length']} | {summary['rejected']} | {summary['timed_out']} | "
            f"{summary['avalanche']} | {summary['config_hash']} | {path} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--check", action="store_true", help="小规模模拟自检，不写入")
    args = parser.parse_args(argv)
    recipe = json.loads(args.config.read_text(encoding="utf-8"))
    print(run_grid(recipe, args.runs_dir, check=args.check))


if __name__ == "__main__":
    main()

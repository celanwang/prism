"""解析模型: scan resident memory and send bytes; no real hardware training."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from prism.env import collect_env, env_fingerprint
from prism.measure import Measurement
from prism.parallel_models import MODE, ModelSpec, Precision, Strategy, predict
from prism.record import load_run, save_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p4_models.json"


def config_hash(config: dict[str, Any]) -> str:
    """Return save_run-compatible SHA-256 of the complete 解析模型 recipe."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scan(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and expand the complete 解析模型 grid before writing anything."""
    if recipe.get("mode") != MODE:
        raise ValueError(f"{MODE}: recipe must explicitly declare its mode")
    axes = ("models", "strategies", "micro_batch_sizes", "micro_batch_counts", "recompute")
    for name in axes:
        if not isinstance(recipe.get(name), list) or not recipe[name]:
            raise ValueError(f"{MODE}: {name} must be a nonempty list")
    precision = Precision(**recipe["precision"])
    rows: list[dict[str, Any]] = []
    for entry, degrees, b, m, recompute in product(*(recipe[name] for name in axes)):
        model = ModelSpec(**entry["spec"])
        strategy = Strategy(**degrees, micro_batch=b, micro_batches=m, recompute=recompute)
        rows.append(
            {
                "mode": MODE,
                "model_name": entry["name"],
                "provenance": entry["provenance"],
                "model": asdict(model),
                "strategy": asdict(strategy),
                "precision": asdict(precision),
                "prediction": predict(
                    model,
                    strategy,
                    precision,
                    compute_flops_per_s=recipe["hypothetical_compute_flops_per_s"],
                ),
            }
        )
    return rows


def render_table(recipe: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Render 解析模型 predictions in bytes and seconds with explicit boundaries."""
    lines = [
        f"{MODE} — P4-T1 显存–通信扫描（非实测）",
        f"配置 SHA-256: {config_hash(recipe)}",
        f"宿主: {platform.platform()}; 解析求值，无训练设备。预热0，求值1次。",
        recipe["rate_provenance"],
        recipe["record_semantics"],
        "显存单位 bytes：理想均衡常驻状态 + 1F1B 最坏 stage 保留激活；不是峰值显存。",
        "通信单位 bytes/optimizer step/device：仅发送；P2P 为 stage 均值，逐 stage 见 JSON。",
        "z>=2 每 microbatch reduce-scatter；z=3 每 microbatch 前后向分别 gather。",
        "N=d*b*m*S；固定 b 扫描时全局 batch 随 d,m 改变。所有数值均为解析模型。",
        "",
        "| 模型 | d/t/p/z | b/m/重算 | 参数 B | 梯度 B | Adam B | 激活 B | 合计 B | "
        "AR B/step | AG B/step | RS B/step | P2P B/step | bubble | 计算 s/step | token/s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        st, out = row["strategy"], row["prediction"]
        mem, comm = out["memory_bytes"], out["communication_send_bytes_per_step"]
        degrees = "/".join(str(st[key]) for key in ("dp", "tp", "pp", "zero"))
        batch = f"{st['micro_batch']}/{st['micro_batches']}/{st['recompute']}"
        values = [
            mem[key] for key in ("parameters", "gradients", "optimizer", "activations", "total")
        ]
        values += [comm[key] for key in ("all_reduce", "all_gather", "reduce_scatter", "p2p")]
        values += [
            out[key]
            for key in ("bubble_fraction", "compute_only_step_s", "compute_only_tokens_per_s")
        ]
        lines.append(
            f"| {row['model_name']} | {degrees} | {batch} | "
            + " | ".join(f"{value:.8g}" for value in values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def save_scan(recipe: dict[str, Any], rows: list[dict[str, Any]], runs_dir: Path) -> Path:
    """Save 解析模型 via save_run, retaining reference-point compatibility metadata.

    The fixed Measurement schema holds row 0's deterministic compute prediction,
    not host timings or empirical training quantiles. Memory fields are unmeasured.
    Full resident memory and communication predictions are in predictions.json.
    """
    config = {
        **recipe,
        "measurement_reference_row": 0,
        "measurement_semantics": f"{MODE}: row 0 compute-only step ms; no timing samples; "
        "p50=p95=p99 are the same deterministic prediction; memory fields unmeasured (0)",
    }
    ms = rows[0]["prediction"]["compute_only_step_s"] * 1000
    measurement = Measurement(
        samples_ms=[ms],
        mean=ms,
        stdev=0.0,
        p50=ms,
        p95=ms,
        p99=ms,
        peak_tracemalloc_bytes=0,
        max_rss_bytes=0,
        config_hash=config_hash(config),
        env_fingerprint=env_fingerprint(collect_env()),
        captured_at=datetime.now(UTC).isoformat(),
    )
    slug = "p4-models-" + datetime.now(UTC).strftime("%H%M%S%f")
    path = save_run(runs_dir, slug, config, measurement)
    payload = {"mode": MODE, "config_hash": config_hash(config), "rows": rows}
    (path / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (path / "scan.md").write_text(render_table(recipe, rows), encoding="utf-8")
    load_run(path)
    return path


def main(argv: list[str] | None = None) -> None:
    """Run a saved 解析模型 scan or a deterministic, write-free --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--check", action="store_true", help="解析模型全网格确定性自检，不落盘")
    parser.add_argument("-o", "--output", type=Path, help="解析模型表格输出文件")
    args = parser.parse_args(argv)
    if args.check and args.output is not None:
        parser.error("解析模型: --check is write-free; cannot combine with -o")
    recipe = json.loads(args.config.read_text(encoding="utf-8"))
    rows = scan(recipe)
    table = render_table(recipe, rows)
    if args.check:
        if rows != scan(recipe):
            raise AssertionError(f"{MODE}: deterministic grid check failed")
        print(f"{MODE}: --check passed, {len(rows)} grid points, no artifacts written")
    else:
        path = save_scan(recipe, rows, args.runs_dir)
        if args.output is not None:
            args.output.write_text(table, encoding="utf-8")
        print(table, end="")
        print(f"{MODE}: save_run artifacts: {path}")


if __name__ == "__main__":
    main()

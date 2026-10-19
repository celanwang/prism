"""Benchmark greedy INT8 self-speculation; --check executes without publishing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.quant import QuantizedSUT
    from prism.sut import SUT

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p2_spec.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
K_VALUES = [1, 2, 4, 8]
ARMS = ["fp32", "int8", "spec1", "spec2", "spec4", "spec8"]
RANDOM_MATRIX = {
    "seeds": [42, 20261019],
    "prompt_lengths": [1, 7, 18, 64, 126, 127, 128, 129, 257],
    "new_tokens": 8,
}
LIMITATIONS = (
    "CPU 实测，约 0.5M 同源小模型；草稿仅 block Linear 为 INT8，每次反量化后 fp32 matmul，"
    "没有整数算子加速；激活、KV、embedding/tied head 均为 fp32。"
    "草稿–验证同源导致接受率虚高，不可外推到大模型或异源草稿；不得人为改坏草稿。"
    "逐 token 硬门槛仅覆盖规定矩阵，不承诺全 prompt 的无条件数值等价。"
    "全接受观测点不能确定接受率下降时的真实曲线；解析曲线假设逐位独立、恒定接受概率，"
    "不是人工降质实测。盈亏平衡模型忽略 prefill、终止轮、cache 截断/同步和 Python 开销，"
    "使用代表性前缀的组件 mean，属于乐观成本模型，不要求 D<T。"
    "端到端包含两模型 prefill、真实自回归、验证、同步、截断、日志及最终输出；EOS 关闭，"
    "预算耗尽不出 bonus，终止轮短于 k 时照实记录。"
    "滑窗区间单独统计 fp32 target-only；窗口内收益不外推到窗口外。"
    "measure() 原协议，含 tracemalloc 开销；峰值仅增量 Python 追踪，漏原生内存，"
    "RSS 是进程生命期高水位且模型共存，均不构成模型内存压缩证据。"
    "两个相反顺序批次，无置信区间；T4 为历史经验分辨参照，超过阈值不自动证明显著性。"
)


def validate_config(config: dict[str, Any]) -> None:
    """Keep the correctness matrix fixed; reject invalid workloads before any writes."""
    if set(config) != {
        "task",
        "phase",
        "sut_path",
        "device",
        "threads",
        "k_values",
        "orders",
        "warmup",
        "repeat",
        "prompt_ids",
        "decode_tokens",
        "random",
        "calibration",
    }:
        raise ValueError("invalid P2-T2 recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P2-T2", "P2", "cpu"):
        raise ValueError("requires P2-T2 / P2 / cpu")
    if type(config["threads"]) is not int or config["threads"] != 1:
        raise ValueError("requires one thread")
    if config["k_values"] != K_VALUES or any(type(k) is not int for k in config["k_values"]):
        raise ValueError("k_values must be [1,2,4,8]")
    if config["orders"] != [ARMS, ARMS[::-1]]:
        raise ValueError("orders must counterbalance all six arms")
    if json.dumps(config["random"], sort_keys=True) != json.dumps(RANDOM_MATRIX, sort_keys=True):
        raise ValueError("random correctness matrix must not be relaxed")
    for key, minimum in (("warmup", 0), ("repeat", 1), ("decode_tokens", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    prompt = config["prompt_ids"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
    ):
        raise ValueError("invalid prompt_ids")
    if not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("invalid sut_path")
    if config["calibration"] != {"report": "results/p0/calibration.md", "resolution_ms": 0.092584}:
        raise ValueError("requires original T4 reference")


def verify_correctness(target: SUT, draft: QuantizedSUT, path: Path) -> dict[str, Any]:
    """Fail immediately on any token disagreement; retain every step for independent audit."""
    import numpy as np

    from prism.speculative import generate_speculative
    from prism.sut import check_sut

    check_sut(path)
    fixture = json.loads((path / "reference_greedy.json").read_text())
    cases = []
    for name, case in fixture["cases"].items():
        cases.append((name, case["prompt_ids"], case["max_new_tokens"], case["token_ids"]))
    for seed in RANDOM_MATRIX["seeds"]:
        for length in RANDOM_MATRIX["prompt_lengths"]:
            prompt = (
                np.random.default_rng(seed).integers(target.config.vocab_size, size=length).tolist()
            )
            expected = target.generate(prompt, max_new_tokens=RANDOM_MATRIX["new_tokens"])
            cases.append(
                (f"random-{seed}-length{length}", prompt, RANDOM_MATRIX["new_tokens"], expected)
            )
    records = []
    for name, prompt, budget, expected in cases:
        cached = target.generate(prompt, max_new_tokens=budget, use_cache=True)
        if cached != expected:
            raise RuntimeError(f"HARD FAIL: fp32 cached divergence: {name}; stop and report")
        for k in K_VALUES:
            result = generate_speculative(target, draft, prompt, max_new_tokens=budget, k=k)
            if result.token_ids != expected:
                first = next(
                    (
                        i
                        for i in range(max(len(result.token_ids), len(expected)))
                        if i >= min(len(result.token_ids), len(expected))
                        or result.token_ids[i] != expected[i]
                    ),
                    None,
                )
                raise RuntimeError(f"HARD FAIL: {name}, k={k}, token={first}; stop and report")
            records.append(
                {
                    "case": name,
                    "k": k,
                    "prompt_ids": prompt,
                    "max_new_tokens": budget,
                    "token_ids": result.token_ids,
                    "passed": True,
                    "statistics": result.statistics(k),
                    "steps": [asdict(s) for s in result.steps],
                }
            )
    return {"passed": True, "matrix_cases": len(records), "cases": records}


def break_even(k: int, draft_ms: float, verify_ms: float, target_ms: float) -> dict[str, Any]:
    """Solve sum(alpha**i, i=0..k) > (k*D+V)/T in the optimistic iid model.

    D is the measured mean per draft token in a k-step draft block, V verifies
    pending+k tokens, T forwards one fp32 pending token. Equality is not a gain.
    The returned threshold is an infimum; null means no feasible alpha in [0,1].
    """
    import math

    if (
        type(k) is not int
        or k < 1
        or any(not math.isfinite(v) or v <= 0 for v in (draft_ms, verify_ms, target_ms))
    ):
        raise ValueError("requires k>=1 and finite positive component costs")
    cost_ratio = (k * draft_ms + verify_ms) / target_ms

    def advance(alpha: float) -> float:
        return sum(alpha**i for i in range(k + 1))

    threshold = None
    if cost_ratio < k + 1:
        low, high = 0.0, 1.0
        if cost_ratio < 1:
            threshold = 0.0
        else:
            for _ in range(60):
                middle = (low + high) / 2
                if advance(middle) > cost_ratio:
                    high = middle
                else:
                    low = middle
            threshold = high
    return {
        "k": k,
        "D_ms": draft_ms,
        "V_ms": verify_ms,
        "T_ms": target_ms,
        "D_over_T": draft_ms / target_ms,
        "cost_in_target_tokens": cost_ratio,
        "strict_positive_gain_alpha_infimum": threshold,
        "feasible": threshold is not None,
        "ideal_max_speedup": (k + 1) / cost_ratio,
        "curve": [
            {"alpha": a, "expected_advance": advance(a), "ideal_speedup": advance(a) / cost_ratio}
            for a in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        ],
    }


def run_benchmark(
    config: dict[str, Any],
    *,
    check: bool = False,
    runs_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Remeasure both autoregressive baselines and speculation with identical budgets."""
    validate_config(config)
    # Pin BLAS before importing NumPy in this process; reject already-conflicting settings.
    for key in THREAD_VARIABLES:
        if os.environ.get(key, "1") != "1":
            raise ValueError(f"{key} must be 1")
        os.environ[key] = "1"
    import numpy as np

    from prism.env import collect_env, env_fingerprint
    from prism.kvcache import _forward_cached
    from prism.measure import measure
    from prism.quant import QuantizedSUT
    from prism.record import load_run, save_run
    from prism.speculative import generate_speculative
    from prism.sut import load_sut

    path = ROOT / config["sut_path"]
    target = load_sut(path)
    draft = QuantizedSUT(target, bits=8)
    prompt, budget = config["prompt_ids"], config["decode_tokens"]
    if (
        max(prompt) >= target.config.vocab_size
        or len(prompt) + max(budget, max(K_VALUES)) > target.config.context_length
    ):
        raise ValueError("timed workload and probes must fit the context/vocabulary")
    correctness = verify_correctness(target, draft, path)
    expected = target.generate(prompt, max_new_tokens=budget, use_cache=True)
    workload_traces = {}
    for k in K_VALUES:
        result = generate_speculative(target, draft, prompt, max_new_tokens=budget, k=k)
        if result.token_ids != expected:
            raise RuntimeError(f"HARD FAIL: benchmark prompt, k={k}; stop and report")
        workload_traces[str(k)] = {
            "token_ids": result.token_ids,
            "statistics": result.statistics(k),
            "steps": [asdict(step) for step in result.steps],
        }
    runtime = {
        "cpu_model": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor() or platform.machine(),
        "device": "cpu",
        "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
    }
    env = collect_env()
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/prism/speculative.py",
            "scripts/spec_bench.py",
            "src/prism/quant.py",
            "src/prism/sut.py",
            "src/prism/kvcache.py",
            "src/prism/measure.py",
            "src/prism/env.py",
            "src/prism/record.py",
        )
    }
    anchors = {
        "task": "P2-T2",
        "phase": "P2",
        "recipe": config,
        "runtime": runtime,
        "sources_sha256": sources,
        "sut_config": asdict(target.config),
        "manifest_sha256": hashlib.sha256((path / "MANIFEST.json").read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256((path / "reference_greedy.json").read_bytes()).hexdigest(),
    }
    session = datetime.now(UTC).strftime("%H%M%S%f")
    records = []

    def record(batch: int, arm: str, component: str, fn: Callable[[], Any]) -> dict[str, Any]:
        record_config = {
            **anchors,
            "batch": batch,
            "order": config["orders"][batch - 1],
            "workload": {
                "arm": arm,
                "component": component,
                "batch_size": 1,
                "eos_id": None,
                "decode_tokens": budget,
                "timing_scope": "end-to-end including prefill"
                if component == "e2e"
                else "fixed first-round prefix; prefill excluded",
            },
        }
        measurement = measure(
            fn, warmup=config["warmup"], repeat=config["repeat"], config=record_config
        )
        run_path = None
        if not check:
            saved = save_run(
                runs_dir or ROOT / "runs",
                f"p2-spec-{session}-b{batch}-{arm}-{component}",
                record_config,
                measurement,
            )
            measurement = load_run(saved).measurement
            run_path = str(saved.relative_to(ROOT)) if saved.is_relative_to(ROOT) else str(saved)
        row = {
            "batch": batch,
            "arm": arm,
            "component": component,
            "path": run_path,
            "measurement": asdict(measurement),
        }
        records.append(row)
        return row

    comparisons, models = [], []
    for batch, order in enumerate(config["orders"], 1):
        for arm in order:
            if arm in ("fp32", "int8"):
                model = target if arm == "fp32" else draft
                record(
                    batch,
                    arm,
                    "e2e",
                    lambda model=model: model.generate(
                        prompt, max_new_tokens=budget, use_cache=True
                    ),
                )
            else:
                k = int(arm[4:])
                record(
                    batch,
                    arm,
                    "e2e",
                    lambda k=k: generate_speculative(
                        target, draft, prompt, max_new_tokens=budget, k=k
                    ),
                )
        arms = {
            r["arm"]: r["measurement"]
            for r in records
            if r["batch"] == batch and r["component"] == "e2e"
        }
        for k in K_VALUES:
            for baseline in ("fp32", "int8"):
                delta = arms[baseline]["p50"] - arms[f"spec{k}"]["p50"]
                comparisons.append(
                    {
                        "batch": batch,
                        "k": k,
                        "baseline": baseline,
                        "p50_speedup": arms[baseline]["p50"] / arms[f"spec{k}"]["p50"],
                        "baseline_minus_spec_p50_ms": delta,
                        "T4": "不可分辨"
                        if abs(delta) < config["calibration"]["resolution_ms"]
                        else "超过 T4 经验参照；不自动证明显著性",
                    }
                )
        # Immutable prefix caches: probes replay genuine INT8 candidates, never corrupt weights.
        prefix = np.array([prompt[:-1]], dtype=np.int64)
        target_past = draft_past = None
        if prefix.shape[1]:
            _, target_past = _forward_cached(target, prefix)
            _, draft_past = _forward_cached(draft, prefix)
        pending = np.array([[prompt[-1]]], dtype=np.int64)
        t = record(
            batch, "fp32", "target-token", partial(_forward_cached, target, pending, target_past)
        )
        for k in K_VALUES:

            def draft_block(
                k: int = k, draft_past: Any = draft_past, pending: Any = pending
            ) -> Any:
                cache, token, candidates = draft_past, pending, []
                for _ in range(k):
                    logits, cache = _forward_cached(draft, token, cache)
                    following = int(logits[0, -1].argmax())
                    candidates.append(following)
                    token = np.array([[following]], dtype=np.int64)
                return candidates, cache

            candidates, last_cache = draft_block()
            verify_tokens = np.array([[prompt[-1], *candidates]], dtype=np.int64)
            d = record(batch, f"spec{k}", "draft-block", draft_block)
            v = record(
                batch,
                f"spec{k}",
                "verify-block",
                partial(_forward_cached, target, verify_tokens, target_past),
            )
            record(
                batch,
                f"spec{k}",
                "draft-sync",
                partial(
                    _forward_cached, draft, np.array([[candidates[-1]]], dtype=np.int64), last_cache
                ),
            )
            models.append(
                {
                    "batch": batch,
                    **break_even(
                        k,
                        d["measurement"]["mean"] / k,
                        v["measurement"]["mean"],
                        t["measurement"]["mean"],
                    ),
                    "observed": workload_traces[str(k)]["statistics"],
                }
            )
    summary = {
        "task": "P2-T2",
        "evidence": "CPU 实测",
        "check_only": check,
        "implementation_passed": True,
        "recipe": config,
        "recipe_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "runtime": runtime,
        "environment": env,
        "env_fingerprint": env_fingerprint(env),
        "anchors": anchors,
        "correctness": correctness,
        "workload_traces": workload_traces,
        "records": records,
        "comparisons": comparisons,
        "analytical_models": models,
        "limitations": LIMITATIONS,
    }
    if not check:
        first = Path(records[0]["path"])
        if not first.is_absolute():
            first = ROOT / first
        (first / "comparison.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        report = report_path or ROOT / "results/p2/speculative.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, Any]) -> str:
    """Keep full environment anchors, raw-run links, observations and analytic curves."""
    config = summary["recipe"]
    rows = [
        "# P2-T2 投机采样报告草稿（待规划层终审）",
        "",
        f"CPU 实测：{summary['runtime']['cpu_model']}；设备 CPU；单线程。"
        f"预热 {config['warmup']} 次，重复 {config['repeat']} 次；"
        "e2e 两个相反顺序批次，组件探针固定顺序。",
        "同一 prompt、生成预算、EOS 关闭，所有基线重新自回归测量。端到端包括 prefill，单位 ms。",
        f"硬门槛：{summary['correctness']['matrix_cases']} 组逐 token 全通过；"
        "fixture 2×4 + 固定种子 2×长度 9×k 4。",
        "",
        f"env 指纹：`{summary['env_fingerprint']}`；配方哈希：`{summary['recipe_hash']}`。",
        "",
        "完整环境与源码/工件锚点：",
        "",
        "```json",
        json.dumps(
            {
                "environment": summary["environment"],
                "runtime": summary["runtime"],
                "sources_sha256": summary["anchors"]["sources_sha256"],
                "manifest_sha256": summary["anchors"]["manifest_sha256"],
                "fixture_sha256": summary["anchors"]["fixture_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 接受率与逐轮推进",
        "",
        "接受率=accepted/proposed（包括拒绝后未检查的候选）；条件接受率=accepted/examined。"
        "逐位给出 proposed/examined/accepted，可区分位置衰减与到达概率。滑窗回退不进入接受率分母。",
        "",
        "| k | accepted/proposed | accepted/examined | 平均投机推进 | 全步骤平均推进 | "
        "回退 tokens | 接受数直方图 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k, trace in summary["workload_traces"].items():
        s = trace["statistics"]
        rows.append(
            f"| {k} | {s['accepted']}/{s['proposed']} | {s['accepted']}/{s['examined']} | "
            f"{s['mean_speculative_advance']:.6f} | {s['mean_all_step_advance']:.6f} | "
            f"{s['fallback_tokens']} | {json.dumps(s['accepted_per_round_histogram'])} |"
        )
    rows.extend(
        [
            "",
            "| k | 位置 | proposed | examined | accepted | 条件接受率 | accepted/proposed |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for k, trace in summary["workload_traces"].items():
        for position in trace["statistics"]["positions"]:
            rows.append(
                f"| {k} | {position['position']} | {position['proposed']} | "
                f"{position['examined']} | {position['accepted']} | "
                f"{position['conditional_rate']} | {position['accepted_per_proposed']} |"
            )
    rows.extend(
        [
            "",
            "硬门槛全矩阵合计（不作性能样本；回退 tokens 单独计数）：",
            "",
            "| k | 接受/提出 | 检查 | 投机轮 | fp32 target-only tokens |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for k in K_VALUES:
        cases = [case["statistics"] for case in summary["correctness"]["cases"] if case["k"] == k]
        counts = {
            key: sum(case[key] for case in cases)
            for key in ("accepted", "proposed", "examined", "speculative_rounds", "fallback_tokens")
        }
        rows.append(
            f"| {k} | {counts['accepted']}/{counts['proposed']} | {counts['examined']} | "
            f"{counts['speculative_rounds']} | {counts['fallback_tokens']} |"
        )
    rows.append("")
    rows.extend(
        [
            "窗口回退的完整矩阵逐位日志、随机 prompts 和输出均保存在 "
            "comparison.json 的 correctness.cases；"
            "定时工作负载日志在 workload_traces。终止轮无 bonus，故平均推进可低于 k+1。",
            "",
            "INT8 单独精度沿用 [P2-T1 报告](quant.md)：两 fixture 及其 "
            "8 个随机 prompt 后缀一致率 1，"
            "reference logits max abs error 0.0651092529；该历史数据仅作精度背景，不作时延对照。",
            "",
            "## 端到端与组件测量",
            "",
            "e2e 是完整生成；target-token/draft-block/verify-block/draft-sync "
            "是同一真实首轮前缀的独立探针，"
            "prefill 排除，不保证组件相加等于 e2e。所有行采用上述硬件/预热/重复协议。",
            "",
            "| batch | arm | component | p50 | p95 | p99 | traced peak bytes | "
            "RSS bytes | config hash | env 指纹 | run |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in summary["records"]:
        m = record["measurement"]
        rows.append(
            f"| {record['batch']} | {record['arm']} | {record['component']} | {m['p50']:.6f} | "
            f"{m['p95']:.6f} | {m['p99']:.6f} | {m['peak_tracemalloc_bytes']} | "
            f"{m['max_rss_bytes']} | "
            f"{m['config_hash']} | {m['env_fingerprint']} | {record['path']} |"
        )
    rows.extend(
        [
            "",
            "## 收益与 T4",
            "",
            "T4=0.092584 ms，来源 results/p0/calibration.md；低于阈值不作收益证据。",
            "",
            "| batch | k | baseline | p50 speedup | baseline-spec p50 ms | T4 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for c in summary["comparisons"]:
        rows.append(
            f"| {c['batch']} | {c['k']} | {c['baseline']} | {c['p50_speedup']:.6f} | "
            f"{c['baseline_minus_spec_p50_ms']:.6f} | {c['T4']} |"
        )
    rows.extend(
        [
            "",
            "## 解析阈值与接受率–收益曲线",
            "",
            "独立恒定接受概率 α 下，E[前进]=Σ(i=0..k) α^i，乐观成本 C=kD+V，"
            "speedup=T·E[前进]/C；收益严格为正需 E[前进]>C/T。D 为 draft-block mean/k，"
            "V 为 verify-block mean，T 为 target-token mean。同一前缀组件实测校准；不假设 D<T。"
            "若 C/T≥k+1，则即使 α=1 也无可行盈亏平衡点；null 表示无可行阈值。"
            "模型忽略同步等额外开销，因此可行阈值也不保证端到端加速。",
            "",
            "| batch | k | D ms | V ms | T ms | D/T | C/T | α 阈值下确界 | 理想最高 speedup |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for m in summary["analytical_models"]:
        rows.append(
            f"| {m['batch']} | {m['k']} | {m['D_ms']:.6f} | {m['V_ms']:.6f} | {m['T_ms']:.6f} | "
            f"{m['D_over_T']:.6f} | {m['cost_in_target_tokens']:.6f} | "
            f"{m['strict_positive_gain_alpha_infimum']} | {m['ideal_max_speedup']:.6f} |"
        )
    rows.extend(
        [
            "",
            "解析曲线数值（非实测；实测接受率观测点见前表，收益见 T4 表）：",
            "",
            "| batch | k | α | 解析 E[前进] | 解析 speedup |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for m in summary["analytical_models"]:
        for point in m["curve"]:
            rows.append(
                f"| {m['batch']} | {m['k']} | {point['alpha']} | {point['expected_advance']:.6f} | "
                f"{point['ideal_speedup']:.6f} |"
            )
    all_accepted = all(
        t["statistics"]["acceptance_rate"] == 1 for t in summary["workload_traces"].values()
    )
    gains = [c for c in summary["comparisons"] if c["baseline"] == "fp32" and c["p50_speedup"] > 1]
    rows.extend(
        [
            "",
            "## 观测与边界",
            "",
            (
                "本工作负载全部接受率为 1；无衰减观测，不人为改坏草稿。"
                if all_accepted
                else "本工作负载出现拒绝，接受率和衰减以逐位日志为准。"
            ),
            (
                "全部 k、两批次相对 fp32 均无 p50 加速，是合法负结果。"
                if not gains
                else "部分点出现 p50 加速，仅限表中硬件、窗口内工作负载及对应批次。"
            ),
            "解析模型的可行性及阈值逐 k 列于前表；不能将高接受率等同于收益。",
            "",
            summary["limitations"],
            "",
            f"逐位日志及完整比较：`{summary['records'][0]['path']}/comparison.json`。",
            "",
            "复现：`uv run --locked python scripts/spec_bench.py`；"
            "`--check` 执行同一完整配方并打印 JSON，不写 runs/ 或 results/。",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    """Run the recipe; publish formal evidence only when --check is absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run_benchmark(config, check=args.check), ensure_ascii=False, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()

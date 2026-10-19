"""Real NumPy GPT training on CPU; calibrate P4-T1 without changing its formulas.

Manual backward retains only its actual saved arrays (no dropout, no checkpointing).
Array storage inventories measure retained bytes, not allocator/device peaks.
measure() separately records traced incremental peak and lifetime RSS reference.
"""

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
CONFIG_PATH = ROOT / "configs/p4_calibrate.json"
THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config: dict) -> None:
    """Validate the entire supported recipe before executing or writing."""
    expected = {
        "task",
        "phase",
        "device",
        "sut_path",
        "sequence_length",
        "micro_batch_sizes",
        "threads",
        "seed",
        "warmup",
        "repeat",
        "optimizer",
        "hypothetical_compute_flops_per_s",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("invalid P4-T2 recipe keys")
    if (config["task"], config["phase"], config["device"]) != ("P4-T2", "P4", "cpu"):
        raise ValueError("requires P4-T2 / P4 / cpu")
    if config["micro_batch_sizes"] != [1, 2, 4] or any(
        type(b) is not int for b in config["micro_batch_sizes"]
    ):
        raise ValueError("requires micro-batch = [1, 2, 4]")
    for key in ("sequence_length", "threads", "seed", "warmup", "repeat"):
        minimum = 0 if key in ("seed", "warmup") else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"invalid {key}")
    if config["threads"] != 1 or not isinstance(config["sut_path"], str) or not config["sut_path"]:
        raise ValueError("requires one native thread and a SUT path")
    optimizer = config["optimizer"]
    if (
        not isinstance(optimizer, dict)
        or set(optimizer)
        != {
            "name",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
        }
        or optimizer["name"] != "AdamW"
    ):
        raise ValueError("requires AdamW recipe")
    for key in ("learning_rate", "beta1", "beta2", "epsilon", "weight_decay"):
        value = optimizer[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"invalid optimizer.{key}")
        if value < 0 or (key in ("learning_rate", "epsilon") and value == 0):
            raise ValueError(f"invalid optimizer.{key}")
        if key in ("beta1", "beta2") and value >= 1:
            raise ValueError(f"invalid optimizer.{key}")
    rate = config["hypothetical_compute_flops_per_s"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid hypothetical compute rate")
    json.dumps(config, allow_nan=False)


def deviation(predicted: float, measured: float) -> dict:
    """Positive delta means overprediction; relative denominator is measurement."""
    if any(not math.isfinite(x) or x < 0 for x in (predicted, measured)):
        raise ValueError("requires finite nonnegative bytes")
    delta = predicted - measured
    return {
        "predicted_bytes": predicted,
        "measured_bytes": measured,
        "signed_delta_bytes": delta,
        "absolute_deviation_bytes": abs(delta),
        "relative_deviation": delta / measured if measured else (0.0 if not predicted else None),
    }


def storage_inventory(arrays: dict) -> list[dict]:
    """Count each live owning ndarray allocation once, including view backing storage."""
    import numpy as np

    owners = {}
    for name, value in arrays.items():
        while isinstance(value.base, np.ndarray):
            value = value.base
        key = id(value)
        if key not in owners:
            owners[key] = {
                "names": [],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "storage_bytes": value.nbytes,
            }
        owners[key]["names"].append(name)
    return list(owners.values())


def inventory_bytes(inventory: list[dict]) -> int:
    return sum(item["storage_bytes"] for item in inventory)


class Trainer:
    """SUT pre-LN causal GPT, tied embeddings, tanh GeLU, CE and FP32 AdamW.

    Weights, gradients and two Adam moments are allocated before measurement.
    No FP32 master copy is needed for FP32 weights. Inputs stay fixed; weights
    advance across diagnostic, warmup and measured steps. No kaleido imports.
    """

    def __init__(self, sut: Any, optimizer: dict) -> None:
        import numpy as np

        self.config = sut.config
        self.params = {name: value.copy() for name, value in sut.arrays.items()}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.first = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.optimizer = optimizer.copy()
        self.steps = 0
        self.last_loss = None

    def _linear(self, x, name):
        return x @ self.params[f"{name}.weight"].T + self.params[f"{name}.bias"]

    def _linear_backward(self, grad, x, name):
        self.grads[f"{name}.weight"] += grad.reshape(-1, grad.shape[-1]).T @ x.reshape(
            -1, x.shape[-1]
        )
        self.grads[f"{name}.bias"] += grad.sum(axis=(0, 1))
        return grad @ self.params[f"{name}.weight"]

    def _norm(self, x, name):
        import numpy as np

        centered = x - x.mean(axis=-1, keepdims=True)
        inverse = np.reciprocal(np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5))
        normalized = centered * inverse
        return normalized * self.params[f"{name}.weight"] + self.params[f"{name}.bias"], {
            "normalized": normalized,
            "inverse_std": inverse,
        }

    def _norm_backward(self, grad, saved, name):
        x, inverse = saved["normalized"], saved["inverse_std"]
        self.grads[f"{name}.weight"] += (grad * x).sum(axis=(0, 1))
        self.grads[f"{name}.bias"] += grad.sum(axis=(0, 1))
        g = grad * self.params[f"{name}.weight"]
        return inverse * (
            g - g.mean(axis=-1, keepdims=True) - x * (g * x).mean(axis=-1, keepdims=True)
        )

    def forward(self, tokens):
        """Return logits and precisely the arrays used by manual backward."""
        import numpy as np

        b, s = tokens.shape
        h, heads = self.config.width, self.config.heads
        d = h // heads
        x = (
            self.params["token_embedding.weight"][tokens]
            + self.params["position_embedding.weight"][:s]
        )
        visible = np.arange(s)[None, :] <= np.arange(s)[:, None]
        layers = []
        for layer in range(self.config.layers):
            prefix = f"blocks.{layer}"
            n1, ln1 = self._norm(x, f"{prefix}.ln1")
            qkv = self._linear(n1, f"{prefix}.attention.qkv")
            q, k, v = (
                part.reshape(b, s, heads, d).transpose(0, 2, 1, 3)
                for part in np.split(qkv, 3, axis=-1)
            )
            scores = q @ k.swapaxes(-2, -1) / np.float32(math.sqrt(d))
            scores = np.where(visible, scores, -np.inf)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            attended = (probability @ v).transpose(0, 2, 1, 3).reshape(b, s, h)
            x = x + self._linear(attended, f"{prefix}.attention.output")
            n2, ln2 = self._norm(x, f"{prefix}.ln2")
            pre = self._linear(n2, f"{prefix}.up")
            gelu = (
                0.5
                * pre
                * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (pre + 0.044715 * pre**3)))
            )
            x = x + self._linear(gelu, f"{prefix}.down")
            layers.append(
                {
                    "n1": n1,
                    "ln1": ln1,
                    "q": q,
                    "k": k,
                    "v": v,
                    "probability": probability,
                    "attended": attended,
                    "n2": n2,
                    "ln2": ln2,
                    "pre_gelu": pre,
                    "gelu": gelu,
                }
            )
        output, norm = self._norm(x, "final_norm")
        return output @ self.params["token_embedding.weight"].T, {
            "layers": layers,
            "final_norm": norm,
            "head_input": output,
        }

    @staticmethod
    def cross_entropy(logits, targets):
        """Stable mean CE and its derivative, with FP32 logits storage."""
        import numpy as np

        flat = logits.reshape(-1, logits.shape[-1])
        shifted = flat - flat.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        sums = exp.sum(axis=-1, keepdims=True)
        rows = np.arange(flat.shape[0])
        loss = float((np.log(sums[:, 0]) - shifted[rows, targets.ravel()]).mean())
        grad = exp / sums
        grad[rows, targets.ravel()] -= 1
        grad /= len(rows)
        return loss, grad.reshape(logits.shape)

    def backward(self, grad, saved, tokens):
        """Propagate CE derivative through every layer and both tied-weight uses."""
        import numpy as np

        for value in self.grads.values():
            value.fill(0)
        self.grads["token_embedding.weight"] += grad.reshape(-1, grad.shape[-1]).T @ saved[
            "head_input"
        ].reshape(-1, self.config.width)
        dx = self._norm_backward(
            grad @ self.params["token_embedding.weight"], saved["final_norm"], "final_norm"
        )
        b, s = tokens.shape
        h, heads = self.config.width, self.config.heads
        d = h // heads
        for layer in reversed(range(self.config.layers)):
            c = saved["layers"][layer]
            prefix = f"blocks.{layer}"
            g = self._linear_backward(dx, c["gelu"], f"{prefix}.down")
            pre = c["pre_gelu"]
            coefficient = np.float32(math.sqrt(2 / math.pi))
            tanh = np.tanh(coefficient * (pre + 0.044715 * pre**3))
            g *= 0.5 * (1 + tanh) + 0.5 * pre * (1 - tanh**2) * coefficient * (
                1 + 3 * 0.044715 * pre**2
            )
            g = self._linear_backward(g, c["n2"], f"{prefix}.up")
            dx = dx + self._norm_backward(g, c["ln2"], f"{prefix}.ln2")
            g = self._linear_backward(dx, c["attended"], f"{prefix}.attention.output")
            g = g.reshape(b, s, heads, d).transpose(0, 2, 1, 3)
            dv = c["probability"].swapaxes(-2, -1) @ g
            dp = g @ c["v"].swapaxes(-2, -1)
            ds = c["probability"] * (dp - (dp * c["probability"]).sum(axis=-1, keepdims=True))
            ds /= np.float32(math.sqrt(d))
            dq = ds @ c["k"]
            dk = ds.swapaxes(-2, -1) @ c["q"]
            dqkv = np.concatenate(
                [part.transpose(0, 2, 1, 3).reshape(b, s, h) for part in (dq, dk, dv)], axis=-1
            )
            g = self._linear_backward(dqkv, c["n1"], f"{prefix}.attention.qkv")
            dx = dx + self._norm_backward(g, c["ln1"], f"{prefix}.ln1")
        np.add.at(self.grads["token_embedding.weight"], tokens, dx)
        self.grads["position_embedding.weight"][:s] += dx.sum(axis=0)

    def update(self):
        """Bias-corrected Adam with decoupled weight decay on all parameters."""
        import numpy as np

        self.steps += 1
        opt = self.optimizer
        b1, b2, lr = opt["beta1"], opt["beta2"], opt["learning_rate"]
        for name, parameter in self.params.items():
            grad, first, second = self.grads[name], self.first[name], self.second[name]
            first *= b1
            first += (1 - b1) * grad
            second *= b2
            second += (1 - b2) * grad**2
            direction = (first / (1 - b1**self.steps)) / (
                np.sqrt(second / (1 - b2**self.steps)) + opt["epsilon"]
            )
            parameter *= 1 - lr * opt["weight_decay"]
            parameter -= lr * direction

    def step(self, tokens, targets, *, capture=False):
        """One genuine forward/backward/update; optional untimed storage diagnostic."""
        logits, saved = self.forward(tokens)
        inventory = self.activation_inventory(saved, logits) if capture else None
        loss, grad = self.cross_entropy(logits, targets)
        self.backward(grad, saved, tokens)
        self.update()
        self.last_loss = loss
        return inventory

    @staticmethod
    def activation_inventory(saved, logits):
        def flatten(value, prefix=""):
            arrays = {}
            if isinstance(value, dict):
                for key, item in value.items():
                    arrays.update(flatten(item, f"{prefix}.{key}"))
            elif isinstance(value, list):
                for key, item in enumerate(value):
                    arrays.update(flatten(item, f"{prefix}.{key}"))
            else:
                arrays[prefix] = value
            return arrays

        return {
            "layers": storage_inventory(flatten(saved["layers"])),
            "all_forward_retained_including_logits": storage_inventory(
                {**flatten(saved), "logits": logits}
            ),
        }

    def state_inventory(self):
        return {
            "parameters": storage_inventory(self.params),
            "gradients": storage_inventory(self.grads),
            "optimizer": storage_inventory(
                {
                    **{f"first.{k}": v for k, v in self.first.items()},
                    **{f"second.{k}": v for k, v in self.second.items()},
                }
            ),
        }


def fit_linearity(rows: list[dict]) -> dict:
    """Test proportionality independently of a free-intercept OLS fit."""
    import numpy as np

    if [r["micro_batch"] for r in rows] != [1, 2, 4]:
        raise ValueError("linearity requires three micro-batches 1,2,4")
    x = np.array([1, 2, 4], dtype=float)
    y = np.array([r["measured_bytes"]["activations"] for r in rows], dtype=float)
    p = np.array([r["prediction"]["memory_bytes"]["activations"] for r in rows])
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("requires positive measured activations")
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(3)]), y, rcond=None)[0]
    residuals = y - (slope * x + intercept)
    proportional_residuals = y - y[0] * x
    return {
        "method": "OLS retained_bytes = intercept + slope * micro_batch; plus B=1 anchored test",
        "slope_bytes_per_micro_batch": float(slope),
        "intercept_bytes": float(intercept),
        "residuals_bytes": residuals.tolist(),
        "r_squared": float(1 - (residuals**2).sum() / ((y - y.mean()) ** 2).sum())
        if (y != y.mean()).any()
        else None,
        "b1_anchored_residuals_bytes": proportional_residuals.tolist(),
        "max_relative_proportional_error": float(np.max(np.abs(proportional_residuals) / y)),
        "predicted_slope_bytes_per_micro_batch": float(p[0]),
        "prediction_minus_measured_slope_bytes": float(p[0] - slope),
        "passes_proportionality_1e_minus_6": bool(
            np.max(np.abs(proportional_residuals) / y) <= 1e-6
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_calibration(recipe: dict, runs_dir: Path, *, check=False) -> dict:
    """Persist each real measurement and its independent analytic comparison."""
    validate_config(recipe)
    recipe = deepcopy(recipe)
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    # CLI imports NumPy and BLAS only after setting the native thread environment.
    import numpy as np

    from prism.measure import measure
    from prism.parallel_models import ModelSpec, Precision, Strategy, predict
    from prism.record import load_run, save_run
    from prism.sut import load_sut

    if check:
        recipe.update(sequence_length=4, warmup=0, repeat=2)
    sut_path = ROOT / recipe["sut_path"]
    sut = load_sut(sut_path)
    if recipe["sequence_length"] > sut.config.context_length or sut.config.dropout != 0:
        raise ValueError("requires context within SUT capacity and dropout=0")
    model = ModelSpec(
        layers=sut.config.layers,
        hidden=sut.config.width,
        ffn=4 * sut.config.width,
        vocab=sut.config.vocab_size,
        seq=recipe["sequence_length"],
        heads=sut.config.heads,
        position_capacity=sut.config.context_length,
    )
    precision = Precision(
        weight_bytes=4, gradient_bytes=4, optimizer_bytes=8, activation_bytes=4, mask_bytes=1
    )
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
        "task": "P4-T2",
        "phase": "P4",
        "recipe": recipe,
        "runtime": runtime,
        "model": asdict(model),
        "precision": asdict(precision),
        "implementation": "prism NumPy manual backward + AdamW; actual SUT weights; dropout=0",
        "evidence": "smoke only; 非实验证据" if check else "单机 CPU 真实训练实测",
        "timing_scope": "forward + mean CE + backward + AdamW update + temporary disposal; "
        "fixed seeded synthetic tokens; states preallocated; diagnostic and setup excluded",
        "memory_scope": "unique ndarray storage at end of forward: layer saved arrays vs "
        "P4 retained layers; logits/final norm separately; tracemalloc incremental peak excludes "
        "preallocated states; RSS is process lifetime high-water reference, not per-step peak",
        "state_evolution": "each B starts from SUT; 1 diagnostic + warmup + repeat updates",
        "sut_manifest_sha256": hashlib.sha256(
            (sut_path / "MANIFEST.json").read_bytes()
        ).hexdigest(),
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/p4_calibrate.py",
                "src/prism/parallel_models.py",
                "src/prism/sut.py",
                "src/prism/measure.py",
                "src/prism/record.py",
            )
        },
    }
    session = datetime.now(UTC).strftime("%H%M%S%f") + ("-check" if check else "")
    rows = []
    for batch in recipe["micro_batch_sizes"]:
        trainer = Trainer(sut, recipe["optimizer"])
        if sum(p.size for p in trainer.params.values()) != model.parameters:
            raise AssertionError("SUT parameter count disagrees with P4-T1; stop")
        rng = np.random.default_rng(recipe["seed"])
        sequence = rng.integers(0, model.vocab, size=(batch, model.seq + 1), dtype=np.int64)
        tokens, targets = sequence[:, :-1], sequence[:, 1:]
        strategy = Strategy(micro_batch=batch)
        prediction = predict(
            model,
            strategy,
            precision,
            compute_flops_per_s=recipe["hypothetical_compute_flops_per_s"],
        )
        record_config = {**common, "strategy": asdict(strategy)}
        activations = trainer.step(tokens, targets, capture=True)
        diagnostic_loss = trainer.last_loss
        states = trainer.state_inventory()
        observed = {key: inventory_bytes(value) for key, value in states.items()}
        observed["activations"] = inventory_bytes(activations["layers"])
        observed["model_states"] = sum(observed[key] for key in states)
        observed["total"] = observed["model_states"] + observed["activations"]
        observed["all_forward_retained_including_logits"] = inventory_bytes(
            activations["all_forward_retained_including_logits"]
        )
        measurement = measure(
            lambda trainer=trainer, tokens=tokens, targets=targets: trainer.step(tokens, targets),
            warmup=recipe["warmup"],
            repeat=recipe["repeat"],
            config=record_config,
        )
        if not math.isfinite(trainer.last_loss) or not all(
            np.isfinite(v).all()
            for group in (trainer.params, trainer.grads, trainer.first, trainer.second)
            for v in group.values()
        ):
            raise FloatingPointError("nonfinite training state")
        if trainer.steps != 1 + recipe["warmup"] + recipe["repeat"]:
            raise AssertionError("missing training updates")
        path = save_run(runs_dir, f"p4-calibrate-{session}-b{batch}", record_config, measurement)
        run = load_run(path)
        row = {
            "micro_batch": batch,
            "path": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path.resolve()),
            "config_hash": measurement.config_hash,
            "prediction": prediction,
            "measured_bytes": observed,
            "deviations": {
                key: deviation(prediction["memory_bytes"][key], observed[key])
                for key in ("parameters", "gradients", "optimizer", "activations", "total")
            },
            "state_inventory": states,
            "activation_inventory": activations,
            "measurement": asdict(run.measurement),
            "diagnostic_loss": diagnostic_loss,
            "last_loss": trainer.last_loss,
            "optimizer_steps": trainer.steps,
            "compute_time_relative_deviation": (
                prediction["compute_only_step_s"] * 1000 - measurement.p50
            )
            / measurement.p50,
            "effective_flops_per_s_at_p50": prediction["training_flops_per_step"]
            / (measurement.p50 / 1000),
        }
        write_json(path / "calibration_point.json", row)
        rows.append(row)
        print(f"saved {path.name}", flush=True)
    summary = {
        **common,
        "recipe_hash": digest(recipe),
        "rows": rows,
        "linearity": fit_linearity(rows),
    }
    first = ROOT / rows[0]["path"]
    write_json(first / "calibration.json", summary)
    (first / "calibration.md").write_text(render_calibration(summary), encoding="utf-8")
    return summary


def render_calibration(summary: dict) -> str:
    """Human-readable full table with provenance and unambiguous byte semantics."""
    recipe = summary["recipe"]
    lines = [
        "# P4-T2 校准数据",
        "",
        summary["evidence"],
        f"硬件/设备：{summary['runtime']['cpu_model']} / CPU；单线程。",
        f"预热 {recipe['warmup']}；重复 {recipe['repeat']}；分位数 NumPy linear。",
        f"配方 SHA-256：`{summary['recipe_hash']}`。",
        "",
        "偏差 Δ=预测−实测；绝对偏差=|Δ|；相对偏差=Δ/实测。单位 bytes。",
        "激活为反向所需层内保留数组；CPU 数组存储实测，不是 GPU 显存或峰值。",
        "",
        "| micro-batch | 项目 | 解析 B | 实测 B | Δ B | 绝对偏差 B | 相对偏差 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        for key in ("parameters", "gradients", "optimizer", "activations", "total"):
            value = row["deviations"][key]
            lines.append(
                f"| {row['micro_batch']} | {key} | {value['predicted_bytes']:.0f} | "
                f"{value['measured_bytes']} | {value['signed_delta_bytes']:+.0f} | "
                f"{value['absolute_deviation_bytes']:.0f} | "
                f"{value['relative_deviation']:+.4%} |"
            )
    lines += [
        "",
        "| B | p50 ms | p95 ms | p99 ms | traced peak B | lifetime RSS B | "
        "全前向保留含 logits B | 假想计算 ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        m = row["measurement"]
        lines.append(
            f"| {row['micro_batch']} | {m['p50']:.6f} | {m['p95']:.6f} | "
            f"{m['p99']:.6f} | {m['peak_tracemalloc_bytes']} | {m['max_rss_bytes']} | "
            f"{row['measured_bytes']['all_forward_retained_including_logits']} | "
            f"{row['prediction']['compute_only_step_s'] * 1000:.6f} |"
        )
    lines += [
        "",
        "## 激活线性性",
        "",
        "```json",
        json.dumps(summary["linearity"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 配置与产物",
        "",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- B={row['micro_batch']}：`{Path(row['path']).name}`；"
            f"config SHA-256 `{row['config_hash']}`；"
            f"诊断/末步 loss={row['diagnostic_loss']:.6f}/{row['last_loss']:.6f}。"
        )
    lines += [
        "",
        "## 边界声明",
        "",
        "单机 CPU、无多卡；通信量未实测，仅解析值（本配置 DP=TP=PP=1，发送量为 0）。",
        "1 TFLOP/s 为配置中的假想计算速率，非硬件标称或训练预测。",
        "tracemalloc 不覆盖所有 native 分配；RSS 是进程历史峰值，不能按 B 相减。",
        "常驻数组在 tracing 前预分配；traced peak 不包含这些常驻状态。",
        "与 P4-T1 对照使用 FP32 权重/梯度/激活、8 B/参数 Adam moments；",
        "P4-T1 默认 12 B/参数另含 master copy，本 FP32 实现不需要该副本。",
        "解析激活仍含 dropout masks 的保守近似；SUT dropout=0。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--check", action="store_true", help="S=4, warmup=0, repeat=2; smoke only")
    parser.add_argument("-o", "--output", type=Path, help="write the full calibration table")
    args = parser.parse_args(argv)
    if args.check and args.output is not None:
        parser.error("--check smoke cannot overwrite an -o evidence report")
    recipe = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_calibration(recipe, args.runs_dir, check=args.check)
    report = render_calibration(summary)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"calibration.json: {Path(summary['rows'][0]['path']) / 'calibration.json'}")


if __name__ == "__main__":
    main()

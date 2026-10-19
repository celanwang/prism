# P4-T2 执行汇报（-o 文件）

任务规格：`tasks/p4-t2-calibrate.md`；分支：`codex/p4-calibrate`。
实现选择：prism 自有 NumPy 手工反向与 AdamW，实际 SUT 权重，478720 参数；不使用记录回放。

## 1. 实际改动/新增文件清单

共 78 个新增文件；全部在白名单内，没有修改既有文件，没有删除文件。
主实现/配方/测试/阶段报告各一份；正式、smoke 与历史产物均明确标记。
续跑修复 JSON 重载后 Markdown 顺序不稳定问题；旧源码快照与早期记录一起保留。

- `configs/p4_calibrate.json`
- `results/p4/REPORT.md`
- `runs/20261019-p4-calibrate-102056043664-b1/calibration.json`
- `runs/20261019-p4-calibrate-102056043664-b1/calibration.md`
- `runs/20261019-p4-calibrate-102056043664-b1/calibration_point.json`
- `runs/20261019-p4-calibrate-102056043664-b1/config.json`
- `runs/20261019-p4-calibrate-102056043664-b1/env.lock`
- `runs/20261019-p4-calibrate-102056043664-b1/measurement.json`
- `runs/20261019-p4-calibrate-102056043664-b1/meta.json`
- `runs/20261019-p4-calibrate-102056043664-b1/source_snapshot.py`
- `runs/20261019-p4-calibrate-102056043664-b2/calibration_point.json`
- `runs/20261019-p4-calibrate-102056043664-b2/config.json`
- `runs/20261019-p4-calibrate-102056043664-b2/env.lock`
- `runs/20261019-p4-calibrate-102056043664-b2/measurement.json`
- `runs/20261019-p4-calibrate-102056043664-b2/meta.json`
- `runs/20261019-p4-calibrate-102056043664-b4/calibration_point.json`
- `runs/20261019-p4-calibrate-102056043664-b4/config.json`
- `runs/20261019-p4-calibrate-102056043664-b4/env.lock`
- `runs/20261019-p4-calibrate-102056043664-b4/measurement.json`
- `runs/20261019-p4-calibrate-102056043664-b4/meta.json`
- `runs/20261019-p4-calibrate-102330548159-check-b1/calibration.json`
- `runs/20261019-p4-calibrate-102330548159-check-b1/calibration.md`
- `runs/20261019-p4-calibrate-102330548159-check-b1/calibration_point.json`
- `runs/20261019-p4-calibrate-102330548159-check-b1/config.json`
- `runs/20261019-p4-calibrate-102330548159-check-b1/env.lock`
- `runs/20261019-p4-calibrate-102330548159-check-b1/measurement.json`
- `runs/20261019-p4-calibrate-102330548159-check-b1/meta.json`
- `runs/20261019-p4-calibrate-102330548159-check-b2/calibration_point.json`
- `runs/20261019-p4-calibrate-102330548159-check-b2/config.json`
- `runs/20261019-p4-calibrate-102330548159-check-b2/env.lock`
- `runs/20261019-p4-calibrate-102330548159-check-b2/measurement.json`
- `runs/20261019-p4-calibrate-102330548159-check-b2/meta.json`
- `runs/20261019-p4-calibrate-102330548159-check-b4/calibration_point.json`
- `runs/20261019-p4-calibrate-102330548159-check-b4/config.json`
- `runs/20261019-p4-calibrate-102330548159-check-b4/env.lock`
- `runs/20261019-p4-calibrate-102330548159-check-b4/measurement.json`
- `runs/20261019-p4-calibrate-102330548159-check-b4/meta.json`
- `runs/20261019-p4-calibrate-114418775926-b1/calibration.json`
- `runs/20261019-p4-calibrate-114418775926-b1/calibration.md`
- `runs/20261019-p4-calibrate-114418775926-b1/calibration_point.json`
- `runs/20261019-p4-calibrate-114418775926-b1/config.json`
- `runs/20261019-p4-calibrate-114418775926-b1/env.lock`
- `runs/20261019-p4-calibrate-114418775926-b1/measurement.json`
- `runs/20261019-p4-calibrate-114418775926-b1/meta.json`
- `runs/20261019-p4-calibrate-114418775926-b2/calibration_point.json`
- `runs/20261019-p4-calibrate-114418775926-b2/config.json`
- `runs/20261019-p4-calibrate-114418775926-b2/env.lock`
- `runs/20261019-p4-calibrate-114418775926-b2/measurement.json`
- `runs/20261019-p4-calibrate-114418775926-b2/meta.json`
- `runs/20261019-p4-calibrate-114418775926-b4/calibration_point.json`
- `runs/20261019-p4-calibrate-114418775926-b4/config.json`
- `runs/20261019-p4-calibrate-114418775926-b4/env.lock`
- `runs/20261019-p4-calibrate-114418775926-b4/measurement.json`
- `runs/20261019-p4-calibrate-114418775926-b4/meta.json`
- `runs/20261019-p4-calibrate-114509446491-check-b1/calibration.json`
- `runs/20261019-p4-calibrate-114509446491-check-b1/calibration.md`
- `runs/20261019-p4-calibrate-114509446491-check-b1/calibration_point.json`
- `runs/20261019-p4-calibrate-114509446491-check-b1/config.json`
- `runs/20261019-p4-calibrate-114509446491-check-b1/env.lock`
- `runs/20261019-p4-calibrate-114509446491-check-b1/measurement.json`
- `runs/20261019-p4-calibrate-114509446491-check-b1/meta.json`
- `runs/20261019-p4-calibrate-114509446491-check-b2/calibration_point.json`
- `runs/20261019-p4-calibrate-114509446491-check-b2/config.json`
- `runs/20261019-p4-calibrate-114509446491-check-b2/env.lock`
- `runs/20261019-p4-calibrate-114509446491-check-b2/measurement.json`
- `runs/20261019-p4-calibrate-114509446491-check-b2/meta.json`
- `runs/20261019-p4-calibrate-114509446491-check-b4/calibration_point.json`
- `runs/20261019-p4-calibrate-114509446491-check-b4/config.json`
- `runs/20261019-p4-calibrate-114509446491-check-b4/env.lock`
- `runs/20261019-p4-calibrate-114509446491-check-b4/measurement.json`
- `runs/20261019-p4-calibrate-114509446491-check-b4/meta.json`
- `runs/20261019-p4-calibrate-delivery/acceptance.json`
- `runs/20261019-p4-calibrate-delivery/acceptance.log`
- `runs/20261019-p4-calibrate-delivery/calibration.md`
- `runs/20261019-p4-calibrate-delivery/task-report.md`
- `runs/20261019-p4-calibrate-delivery/verification.json`
- `scripts/p4_calibrate.py`
- `tests/test_p4_calibrate.py`

## 2. 每条验收命令输出摘要

| 验收命令 | 输出摘要 | 退出码 |
| --- | --- | ---: |
| `uv sync --locked` | Resolved 9 packages in 2ms；Checked 8 packages in 0.12ms | 0 |
| `uv run --locked pytest` | 477 passed in 6.78s（其中 P4-T2 新增 17 项） | 0 |
| `uv run --locked ruff check .` | All checks passed! | 0 |
| `uv run --locked ruff format --check .` | 54 files already formatted | 0 |
| `uv run --locked python scripts/p4_calibrate.py --check` | B=1/2/4 三点真实训练 smoke；S=4、预热0、重复2；save/load成功；明确标记非实验证据 | 0 |
| `git status --short` | 仅有白名单新增文件；验收时实现、正式与早期产物已暂存。验收日志、最终 smoke 与汇报随后一并暂存提交 | 0 |

完整逐命令输出在同目录 `acceptance.log`，结构化输出在 `acceptance.json`。
额外核验：git diff --cached --check；12个 run 的 load_run/hash、当前或归档源码哈希、更新步数、
常驻量偏差、正式表重载渲染一致性、报告数字/本地链接及文件白名单均通过，详见 `verification.json`。

## 3. 校准数据全表与偏差分析

### 最终正式训练校准全表

单机 CPU 真实训练实测
硬件/设备：Apple M4 Max / CPU；单线程。
预热 5；重复 30；分位数 NumPy linear。
配方 SHA-256：`7ce69f0b052305620caedce32d8bbe687f47c17b0d6b9541ecc1a9daad2af0ca`。

偏差 Δ=预测−实测；绝对偏差=|Δ|；相对偏差=Δ/实测。单位 bytes。
激活为反向所需层内保留数组；CPU 数组存储实测，不是 GPU 显存或峰值。

| micro-batch | 项目 | 解析 B | 实测 B | Δ B | 绝对偏差 B | 相对偏差 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 1 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 1 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 1 | activations | 3342336 | 2623488 | +718848 | 718848 | +27.4005% |
| 1 | total | 11001856 | 10283008 | +718848 | 718848 | +6.9906% |
| 2 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 2 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 2 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 2 | activations | 6684672 | 5246976 | +1437696 | 1437696 | +27.4005% |
| 2 | total | 14344192 | 12906496 | +1437696 | 1437696 | +11.1393% |
| 4 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 4 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 4 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 4 | activations | 13369344 | 10493952 | +2875392 | 2875392 | +27.4005% |
| 4 | total | 21028864 | 18153472 | +2875392 | 2875392 | +15.8394% |

| B | p50 ms | p95 ms | p99 ms | traced peak B | lifetime RSS B | 全前向保留含 logits B | 假想计算 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.051125 | 5.239298 | 5.353562 | 5847848 | 67649536 | 3017216 | 0.417989 |
| 2 | 7.713604 | 7.983909 | 8.218588 | 11683187 | 109019136 | 6034432 | 0.835977 |
| 4 | 13.280354 | 14.050111 | 14.516120 | 23351243 | 205488128 | 12068864 | 1.671954 |

## 激活线性性

```json
{
  "b1_anchored_residuals_bytes": [
    0.0,
    0.0,
    0.0
  ],
  "intercept_bytes": -1.1177718604895825e-09,
  "max_relative_proportional_error": 0.0,
  "method": "OLS retained_bytes = intercept + slope * micro_batch; plus B=1 anchored test",
  "passes_proportionality_1e_minus_6": true,
  "predicted_slope_bytes_per_micro_batch": 3342336.0,
  "prediction_minus_measured_slope_bytes": 718847.9999999995,
  "r_squared": 1.0,
  "residuals_bytes": [
    4.656612873077393e-10,
    0.0,
    0.0
  ],
  "slope_bytes_per_micro_batch": 2623488.0000000005
}
```

## 配置与产物

- B=1：`20261019-p4-calibrate-114418775926-b1`；config SHA-256 `1c52b92b6262333b47d4ead385059223ba5f9e15e09ab946f69851230120f814`；诊断/末步 loss=12.102966/5.483827。
- B=2：`20261019-p4-calibrate-114418775926-b2`；config SHA-256 `0cd65103748f9b13cb5e2146e1ffa7bf802f74e2b216a86859449d1d37c9d35d`；诊断/末步 loss=12.478195/5.810217。
- B=4：`20261019-p4-calibrate-114418775926-b4`；config SHA-256 `7023c6798f6825b1be749a4cd8ca6beba5fa1e9b109cfd0c3e60d3ff42067290`；诊断/末步 loss=12.259622/6.050737。

## 边界声明

单机 CPU、无多卡；通信量未实测，仅解析值（本配置 DP=TP=PP=1，发送量为 0）。
1 TFLOP/s 为配置中的假想计算速率，非硬件标称或训练预测。
tracemalloc 不覆盖所有 native 分配；RSS 是进程历史峰值，不能按 B 相减。
常驻数组在 tracing 前预分配；traced peak 不包含这些常驻状态。
与 P4-T1 对照使用 FP32 权重/梯度/激活、8 B/参数 Adam moments；
P4-T1 默认 12 B/参数另含 master copy，本 FP32 实现不需要该副本。
解析激活仍含 dropout masks 的保守近似；SUT dropout=0。

偏差分析：常驻状态全部对齐；P4-T1 默认 optimizer 12 B 包含 master copy，本 FP32 实现
以现有 Precision 入口设为 8 B（两份 moments）；否则 optimizer 将高估 1914880 B/50%。
B=1 激活高估 718848 B/27.4005%，可由额外 attention 中间量 524288 B、
dropout masks 196608 B 与实际 norm inverse_std 的 −2048 B 完整解释；B=2/4 同比例扩大。
B=1/2/4 激活除以 B 均为 2623488 B，锚点残差0、OLS R²=1；只验证固定 S 的缓存形状比例。
假想 1 TFLOP/s compute-only 时间相对 p50 偏差为 -91.7248%/-89.1623%/-87.4103%，
未以单点系数替代模型；Python、分配、kernel 利用率和 tracing 的贡献没有分别实测。
单机 CPU；无多卡；通信量未实测，仅解析值。常驻/保留数组、tracemalloc 增量峰值、
生命周期 RSS 口径不同，不把任何一项误写为 GPU 峰值显存。完整原理与边界见 `results/p4/REPORT.md`。

## 4. 与规格的任何偏差及原因

**无。** 采用规格明确允许的自有 NumPy 训练实现；FP32/8 B optimizer 经 P4-T1 已有精度入口配置。
未改动 src/prism、依赖、测量协议或 P4-T1 公式，未 push，未删除文件。
未提供具体 -o 路径，因此将规格第 6 节汇报保存为当前文件；
CLI 的 -o 校准全表另存同目录 `calibration.md`。

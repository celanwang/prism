# P4-T2 校准数据

smoke only; 非实验证据
硬件/设备：Apple M4 Max / CPU；单线程。
预热 0；重复 2；分位数 NumPy linear。
配方 SHA-256：`0076bff47cb00556ecb6db69d65a30ffd039e9858b58996c4130deff0f86bdb7`。

偏差 Δ=预测−实测；绝对偏差=|Δ|；相对偏差=Δ/实测。单位 bytes。
激活为反向所需层内保留数组；CPU 数组存储实测，不是 GPU 显存或峰值。

| micro-batch | 项目 | 解析 B | 实测 B | Δ B | 绝对偏差 B | 相对偏差 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 1 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 1 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 1 | activations | 68736 | 66112 | +2624 | 2624 | +3.9690% |
| 1 | total | 7728256 | 7725632 | +2624 | 2624 | +0.0340% |
| 2 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 2 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 2 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 2 | activations | 137472 | 132224 | +5248 | 5248 | +3.9690% |
| 2 | total | 7796992 | 7791744 | +5248 | 5248 | +0.0674% |
| 4 | parameters | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 4 | gradients | 1914880 | 1914880 | +0 | 0 | +0.0000% |
| 4 | optimizer | 3829760 | 3829760 | +0 | 0 | +0.0000% |
| 4 | activations | 274944 | 264448 | +10496 | 10496 | +3.9690% |
| 4 | total | 7934464 | 7923968 | +10496 | 10496 | +0.1325% |

| B | p50 ms | p95 ms | p99 ms | traced peak B | lifetime RSS B | 全前向保留含 logits B | 假想计算 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.530917 | 2.547304 | 2.548761 | 883928 | 46628864 | 78416 | 0.011538 |
| 2 | 2.489084 | 2.523846 | 2.526936 | 970536 | 56426496 | 156832 | 0.023077 |
| 4 | 2.714187 | 2.736782 | 2.738790 | 1143752 | 64880640 | 313664 | 0.046154 |

## 激活线性性

```json
{
  "b1_anchored_residuals_bytes": [
    0.0,
    0.0,
    0.0
  ],
  "intercept_bytes": -7.621171776065335e-11,
  "max_relative_proportional_error": 0.0,
  "method": "OLS retained_bytes = intercept + slope * micro_batch; plus B=1 anchored test",
  "passes_proportionality_1e_minus_6": true,
  "predicted_slope_bytes_per_micro_batch": 68736.0,
  "prediction_minus_measured_slope_bytes": 2623.999999999971,
  "r_squared": 1.0,
  "residuals_bytes": [
    4.3655745685100555e-11,
    2.9103830456733704e-11,
    -5.820766091346741e-11
  ],
  "slope_bytes_per_micro_batch": 66112.00000000003
}
```

## 配置与产物

- B=1：`20261019-p4-calibrate-114509446491-check-b1`；config SHA-256 `ec43936251a52c0688c924bb0528713b6bf7317aafff5ab05da7ed2b3205aae9`；诊断/末步 loss=10.203890/6.919600。
- B=2：`20261019-p4-calibrate-114509446491-check-b2`；config SHA-256 `1475a63f96d9c06ac0c86080da5798455c13d868afa3717ee09587a69c43f80d`；诊断/末步 loss=10.827696/8.272097。
- B=4：`20261019-p4-calibrate-114509446491-check-b4`；config SHA-256 `2f934a629a8172c08cdbc7b221cf79350493f472a7063c7005d7e601ca6cc780`；诊断/末步 loss=10.572502/8.284252。

## 边界声明

单机 CPU、无多卡；通信量未实测，仅解析值（本配置 DP=TP=PP=1，发送量为 0）。
1 TFLOP/s 为配置中的假想计算速率，非硬件标称或训练预测。
tracemalloc 不覆盖所有 native 分配；RSS 是进程历史峰值，不能按 B 相减。
常驻数组在 tracing 前预分配；traced peak 不包含这些常驻状态。
与 P4-T1 对照使用 FP32 权重/梯度/激活、8 B/参数 Adam moments；
P4-T1 默认 12 B/参数另含 master copy，本 FP32 实现不需要该副本。
解析激活仍含 dropout masks 的保守近似；SUT dropout=0。

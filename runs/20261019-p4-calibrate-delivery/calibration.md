# P4-T2 校准数据

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

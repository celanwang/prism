# P2-T3：算子级优化逐项消融（报告草稿）

依据 `tasks/p2-t3-kernels.md`。三项为独立的单开关配方，分别调用
`save_run()`；没有组合优化的测量。CPU 实测，未使用 MPS/CUDA。

## 实现与正确性

- 基线：`AblatingSUT(..., "baseline")`，复现原始合并 QKV、heads-first 视图、
  分配式 MLP；测试独立对照 `SUT.forward` 与 `_forward_cached`。
- QKV 反向消融：同一权重/偏置按输出维切成三份，三次 GEMM 对照基线一次 GEMM，
  不重排或复制权重。表中的候选是“拆分”；判断合并的价值时需反向解读。
- attention 布局：基线从合并 QKV 上取得带跨步的 heads-first 视图；候选先将
  每份 Q/K/V 整理成紧凑的 batch×time 主序缓冲，再 reshape/transpose 为同一
  matmul 的 head 视图。只改变存储/步长，不替换 score/softmax 算法；复制成本
  在计时内。单 token Q/K/V 片段可能已经连续，整理成为无复制操作；不代表所有
  attention 布局方案的结果。计时采用缓存路径，K/V 仍按基线规则复制或拼接成
  自有缓存，因此该消融的布局差异主要作用在 Q 和创建缓存前的临时缓冲。
- MLP 缓冲：候选用 `out=` 复用 up、GELU 临时量与 down 三个私有缓冲，保持
  原 GELU 运算顺序。只保留最近的 `(batch,time)` 形状，跨层和跨调用复用；
  形状变化重新分配，实例不支持并发重入。输出 logits/cache 不引用这些缓冲。
- 每一变体（含基线）的 `within_window`、`window_rollover` 两个独立 fixture，
  均在缓存与非缓存路径逐 token 全等通过；fixture 文件先由 `check_sut()` 校验
  哈希。测试另覆盖 batch=2、因果性、缓存增量 logits、缓冲所有权和形状切换。

## 负载与测量协议

- 模型：本地 kaleido-p3，float32，2 层，width=128，4 heads，vocab=512，
  context=128；权重和 manifest 沿用完整性检查，原始模块未改动。
- 每项、每臂、每负载：预热 **100** 次、重复 **2000** 次，与 T4 配方一致。
  使用 P0 `measure()`；原始样本、p50/p95/p99（NumPy linear 插值）、内存和环境
  均由 `save_run()` 保存并经 `load_run()` 回读校验。
- 每项两批顺序：B1 为 baseline→variant，B2 为 variant→baseline；每批分别测
  prefill 和 decode，不跨项共享基线的时间样本。每个配方单独启动 Python 进程。
- prefill：batch=1、T=128，NumPy RNG seed=42 的固定 token 序列，包含 KV 创建。
- decode：batch=1、fixture 的 18-token prompt，32 个固定 teacher-forced 缓存步；
  prompt prefill 在计时外。所有候选使用同一基线 continuation，每样本从同一
  初始缓存引用开始，forward 只创建新缓存，不修改初始缓存。无 EOS、无 rollover。
  下列 decode 数字为 **32 步合计 ms**，不是单 token 时延。
- MLP 私有缓冲在预热后已分配，报告衡量稳态；attention 整理在每次 forward 内。
  Python/native 的首次启动和冷缓冲分配成本不由稳态数字表征。

## T4 判定规则与边界

参照 `results/p0/calibration.md` 的经验阈值 **0.092584 ms**。
令 Δ = baseline p50 − variant p50。两批 Δ 均 ≥ 阈值，候选“值得”；
两批 Δ 均 ≤ −阈值，候选“不值得”；任一批绝对差低于阈值或方向不一致，
结论为“不可分辨”。不按重复数或 token 数缩小阈值，不断言性能方向的测试。
“值得/不值得”仅描述本负载、此实现、两批顺序下的观测。

T4 是相同软件/硬件环境下的经验扰动参照，不是统计显著性保证，也不是硬件计时器
分辨率。两批为同进程顺序测量，仍可能受温度、频率和调度漂移影响；未估计置信区间，
未针对本模型的分配行为独立重新标定。tracemalloc 扰动分配、遗漏部分 native 内存；
RSS 为进程生命周期峰值，不能当作该算子的独立内存开销。低于阈值的差异不作为收益。

## 实测数据与结论

下方数据由持久化 runs 回读生成。

正式测量为下列三次独立执行（均退出 0）：

```bash
uv run --locked python scripts/kernel_bench.py --recipe qkv
uv run --locked python scripts/kernel_bench.py --recipe attention
uv run --locked python scripts/kernel_bench.py --recipe mlp
```

硬件：**Apple M4 Max**，CPU / arm64，逻辑 CPU=16，
物理内存=51539607552 bytes；五个 native 线程环境变量均为 1。
环境指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`（24 runs 一致，且与 T4 一致）。

配方文件 sorted JSON SHA-256：
`2fec7e97db4ce6ed43e260435a342fb798d2bd3586d20e7252d151ac49bd93a9`。
每条测量的 config hash 还包含单项配方、arm、批次、实际 token、模型、源文件和 runtime，
因此不同 arm/batch 的 hash 不同；完整映射见追溯表。

### 正式配方的逐项对照

每个单元格依次为 **p50 / p95 / p99，单位 ms**；B1、B2 不合并样本。

| 消融 | 负载 | 批次 | baseline | variant | Δ p50 (baseline−variant) |
| --- | --- | --- | --- | --- | ---: |
| qkv | prefill | B1 | 1.768729 / 2.298485 / 2.561612 | 1.932875 / 2.400557 / 2.747208 | -0.164146 |
| qkv | prefill | B2 | 1.806895 / 2.026760 / 2.136878 | 1.904542 / 2.006477 / 2.108220 | -0.097646 |
| qkv | decode | B1 | 14.242479 / 14.643456 / 14.991768 | 16.024208 / 16.402781 / 17.144198 | -1.781729 |
| qkv | decode | B2 | 14.552063 / 15.314017 / 15.832058 | 16.082125 / 16.566216 / 17.174910 | -1.530062 |
| attention | prefill | B1 | 1.766979 / 2.025602 / 2.076708 | 1.916708 / 2.181994 / 2.336816 | -0.149728 |
| attention | prefill | B2 | 1.956730 / 2.206954 / 2.401147 | 2.087855 / 2.262404 / 2.421971 | -0.131125 |
| attention | decode | B1 | 14.660854 / 15.411578 / 15.669557 | 15.658625 / 16.238727 / 16.570590 | -0.997771 |
| attention | decode | B2 | 15.411333 / 16.225165 / 16.963418 | 15.782167 / 16.542838 / 17.312971 | -0.370834 |
| mlp | prefill | B1 | 1.848708 / 2.278302 / 2.690773 | 1.878458 / 2.277420 / 3.301283 | -0.029750 |
| mlp | prefill | B2 | 1.818958 / 2.161466 / 2.397300 | 1.801916 / 2.180392 / 2.510259 | +0.017042 |
| mlp | decode | B1 | 14.779166 / 15.927645 / 17.354768 | 14.916875 / 15.822595 / 16.722857 | -0.137709 |
| mlp | decode | B2 | 14.846312 / 15.942435 / 17.232293 | 14.840895 / 15.652636 / 16.269380 | +0.005417 |

### 正式配方的 T4 强制结论

| 消融候选 | prefill | decode 32 步 | 原因与限制 |
| --- | --- | --- | --- |
| QKV 拆分 | 不值得 | 不值得 | 两批均慢于合并且超过 T4；因此本轮支持保留合并。prefill B2 仅略超阈值，需结合复跑审视稳定性。 |
| attention BT 整理 | 不值得 | 不值得 | 两批差异均超过 T4。额外整理和 Python/NumPy 调用可能抵消连续布局的好处；未单独测组件，不能将差异全部归因到复制。 |
| MLP 缓冲复用 | 不可分辨 | 不可分辨 | prefill 两批差异均低于阈值；decode 一批较慢超阈值，另一批低于阈值且方向改变，不能证明时延收益或稳定退化。 |

以上均为完整前向负载的观测，不是孤立 GEMM、softmax 或 GELU 的组件时延。
MLP 的 `out=` 可减少临时分配，但更多显式 ufunc 调用也有开销，最终时延需以数据判断。
以下为正式运行的原始结论；包含验收复跑的最终建议见后文。

### Run / config hash 追溯

每项各 8 个 run：2 个负载 × 2 个顺序批次 × 2 个 arm。
每个目录包含 `config.json`、`measurement.json`、`env.lock`、`meta.json`；
每项首个目录另含该项 `comparison.json`（正确性 token 清单、对照和结论）。

| run | config hash（SHA-256） |
| --- | --- |
| [20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill](../../runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/measurement.json) | `b4c101a56f250b48113f3f246698a2467278443e08b635e11818687322c30f07` |
| [20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill](../../runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill/measurement.json) | `86a467199e66a6e77ac1ed7a30149d61241d9a23c42a939d9245d7ff6c1cc5ec` |
| [20261019-p2-kernels-120650066022-qkv-b1-baseline-decode](../../runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-decode/measurement.json) | `33f519d69f75a8acb550f5e0dc144968cbff090592f244087a759adcdcf6146f` |
| [20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode](../../runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode/measurement.json) | `a73b36aae9405acab456aa667cb444ada09e90e66683dccd6a4d00de4dd1c43a` |
| [20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill](../../runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill/measurement.json) | `095332344e1d2bc907c71bd26c4011c5e002b3f6478d9df203833610b041aaca` |
| [20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill](../../runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill/measurement.json) | `0df898ef83022fb258870b5030b610d4a23ea8eddf18d3bbefe90fdb288672d2` |
| [20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode](../../runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode/measurement.json) | `266af405094336d34090c1733e90c34f441d04ef9aba8b3a7697fd9529af5683` |
| [20261019-p2-kernels-120650066022-qkv-b2-baseline-decode](../../runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-decode/measurement.json) | `5d396ea2b1b9edd00f511d0f76d160c54cbeeb29a6b2869d6ef213fcc6e70d34` |
| [20261019-p2-kernels-120910325237-attention-b1-baseline-prefill](../../runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/measurement.json) | `a170e031a74c55bfa75a1df4a8e56c93f5584a984639e154d28d7cc7640ccf32` |
| [20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill](../../runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill/measurement.json) | `23755f79fff193ad866c544a8905cbc3bc4e9e9ecab3061442226a80bf4681c0` |
| [20261019-p2-kernels-120910325237-attention-b1-baseline-decode](../../runs/20261019-p2-kernels-120910325237-attention-b1-baseline-decode/measurement.json) | `749ee3b507cc80d2436cdcff4a0d0aa47a650b206887e0d50e11441cc4586624` |
| [20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode](../../runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode/measurement.json) | `fa3084684391515b15fdf4eb27f5cc38782edc72f8da6784e75f0798f23dd4bf` |
| [20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill](../../runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill/measurement.json) | `728f2f5ef74a58863d5574eaa28bfe225864384edd1522fbf26ae12628e1c5e1` |
| [20261019-p2-kernels-120910325237-attention-b2-baseline-prefill](../../runs/20261019-p2-kernels-120910325237-attention-b2-baseline-prefill/measurement.json) | `31d312269f5d855648ff471b1943e9a7ff803ecd59aabf48d997957a53d8a3c4` |
| [20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode](../../runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode/measurement.json) | `c0ea6e8a0f7476e8068a391437b1a12634c076c31219d3c316804631fdc08638` |
| [20261019-p2-kernels-120910325237-attention-b2-baseline-decode](../../runs/20261019-p2-kernels-120910325237-attention-b2-baseline-decode/measurement.json) | `0249f4a2b42298f7f7dd548ce08ed6721ba8eb7f6e7ea2bcc8b5dce1e489476c` |
| [20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill](../../runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/measurement.json) | `60ce4b682acb32a93ec9b1005a3d8811a93226a7b1ad8d45f8f823805549813b` |
| [20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill](../../runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill/measurement.json) | `c2ad695e31a7de236a99b5effef3afaaaee3d9ba9c29c17091b56ad5538d6b9a` |
| [20261019-p2-kernels-121132154397-mlp-b1-baseline-decode](../../runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-decode/measurement.json) | `b8bc9eabe263626552a6a55fe0d73fa13830a3bcaa48a2d9629f889e8d88c99f` |
| [20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode](../../runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode/measurement.json) | `fe3d41186d61aa7e7d75e6b313e92e4c8bdbdf4f2c497c17b78d077ec84d419f` |
| [20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill](../../runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill/measurement.json) | `962281cb74074e2fd8ad8c17cbc08c09dfae13141120963f7c77fd640ed5460c` |
| [20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill](../../runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill/measurement.json) | `f6a15f1873a199529419f375ce116529dae06fb83a112709753a0ef2c4634656` |
| [20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode](../../runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode/measurement.json) | `e8fa23e357b79601d00edafea7142fea314d86e316252ce9d90554be1d2f7f69` |
| [20261019-p2-kernels-121132154397-mlp-b2-baseline-decode](../../runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-decode/measurement.json) | `bfab51723c301ff28d7861a8cecc10e01f60a21a2eaef42c5e03b6d4ab1c0147` |

## 验收复跑与最终建议

`uv run --locked python scripts/kernel_bench.py --check` 完整执行同样的 warmup=100、repeat=2000，退出 0；
三项依次独立测量，不调用 `save_run()`。其完整 stdout（含 24 组原始样本）保存在
[`acceptance-kernel-check.log`](../../runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/acceptance-kernel-check.log)。
逐条核对：复跑与正式 24 个测量的 config hash、env 指纹完全一致；
正式记录全部通过 `load_run()`、源文件哈希与样本分位数重算校验。
复跑仍在同一环境，不能视作跨硬件/跨日复现。

复跑中的单元格依次为 **p50 / p95 / p99，单位 ms**。

| 消融 | 负载 | 批次 | baseline | variant | Δ p50 |
| --- | --- | --- | --- | --- | ---: |
| qkv | prefill | B1 | 1.776583 / 1.905637 / 1.975703 | 1.889125 / 2.012503 / 2.074250 | -0.112542 |
| qkv | prefill | B2 | 1.791416 / 1.910337 / 1.966504 | 1.877687 / 2.010972 / 2.101728 | -0.086271 |
| qkv | decode | B1 | 14.461229 / 14.716294 / 14.813713 | 16.147458 / 16.428063 / 16.681930 | -1.686229 |
| qkv | decode | B2 | 14.728271 / 15.514891 / 15.907927 | 16.227416 / 16.856783 / 17.475422 | -1.499145 |
| attention | prefill | B1 | 1.840791 / 2.000631 / 2.216381 | 1.901209 / 2.089807 / 2.358920 | -0.060418 |
| attention | prefill | B2 | 1.832583 / 2.014335 / 2.276050 | 1.921562 / 2.226668 / 2.497197 | -0.088979 |
| attention | decode | B1 | 14.542896 / 14.870431 / 15.402402 | 15.065375 / 15.785558 / 16.738657 | -0.522479 |
| attention | decode | B2 | 14.527125 / 14.803640 / 15.153137 | 14.966292 / 15.231552 / 15.495908 | -0.439167 |
| mlp | prefill | B1 | 1.854021 / 1.976963 / 2.053688 | 1.810563 / 1.937719 / 2.011335 | +0.043458 |
| mlp | prefill | B2 | 1.855063 / 1.971517 / 2.061621 | 1.810521 / 1.930446 / 1.978709 | +0.044542 |
| mlp | decode | B1 | 14.527292 / 14.800753 / 14.965528 | 14.933584 / 15.172598 / 15.372765 | -0.406292 |
| mlp | decode | B2 | 14.497355 / 15.193701 / 15.795317 | 14.955875 / 15.383465 / 15.799004 | -0.458521 |

两次执行各自按同一 T4 规则判定。最终只保留两次都可分辨、且方向一致的结论；
任一次不可分辨或方向不一致，最终建议降级为不可分辨。没有合并样本或重新估计阈值。

| 候选 | 负载 | 正式判定 | 验收复跑判定 | 最终判定 |
| --- | --- | --- | --- | --- | --- |
| qkv | prefill | 不值得 | 不可分辨 | 不可分辨 |
| qkv | decode | 不值得 | 不值得 | 不值得 |
| attention | prefill | 不值得 | 不可分辨 | 不可分辨 |
| attention | decode | 不值得 | 不值得 | 不值得 |
| mlp | prefill | 不可分辨 | 不可分辨 | 不可分辨 |
| mlp | decode | 不可分辨 | 不值得 | 不可分辨 |

- **QKV 合并**：decode 中拆分在两次执行、四个顺序批次均慢于合并且超过 T4，
  本负载值得保留合并。prefill 的正式 B2 差异 −0.097646 ms 仅略超阈值，
  复跑 B2 为 −0.086271 ms，低于阈值，因此 prefill 的合并收益不可分辨。
- **attention BT 整理**：prefill 正式超阈值，复跑两批均低于阈值，最终不可分辨；
  decode 两次都为稳定的超阈值退化，本路径不值得采用。候选仍执行原缓存复制/拼接，
  不能将该结果泛化为所有 time-major attention 算法的结论。
- **MLP 缓冲复用**：prefill 两次均低于阈值；decode 正式不可分辨，复跑显示退化，
  跨执行不足以确认稳定退化，最终仍不可分辨，未证实时延收益。显式 ufunc 调用、
  tracing 对分配的扰动和系统漂移可能影响对照，但没有组件测量支持单一原因归因。

峰值内存的额外限制：traced peak 不包括 tracing 启动前已经分配的私有 scratch，
因此本报告不从这些数字推导总内存收益。

## 验收与规格偏差

第 5 节的 sync、全仓 pytest（533 passed）、ruff check、ruff format --check、
`prism sut check` 和完整 `kernel_bench.py --check` 全部退出 0；
`git status --short` 的白名单审计及完整文件清单见本次执行汇报。
规格偏差：**无**。负结果和不可分辨结果保留，不据此宣布 P2 整阶段完成。

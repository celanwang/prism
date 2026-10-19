# P0 首份基准报告

## P0 阶段验收总览

| 任务 | 交付物与验收证据指针 |
| --- | --- |
| T1 环境锁定 | `src/prism/env.py`、`configs/env.lock`、`tests/test_env.py`；本地合并提交 `86fd087`（无 PR 号） |
| T2 SUT | `src/prism/sut.py`、`data/sut/kaleido-p3/MANIFEST.json`、`tests/test_sut.py`；[PR #2](https://github.com/celanwang/prism/pull/2) |
| T3 测量协议 | `src/prism/measure.py`、`tests/test_measure.py`；[PR #1](https://github.com/celanwang/prism/pull/1) |
| T4 标定 | [标定报告](calibration.md)、`configs/calibrate.json`、`tests/test_calibrate.py`；[PR #4](https://github.com/celanwang/prism/pull/4) |
| T5 记录 | `src/prism/record.py`、`tests/test_record.py`；[PR #3](https://github.com/celanwang/prism/pull/3) |
| T6 基准闭环 | `scripts/baseline.py`、`configs/baseline.json`、`tests/test_baseline.py`、本报告及下列 runs；当前分支交付，尚无 PR |

T1–T5 已合并；T6 提供实测证据和复现入口，P0 最终验收由规划层独立重跑后决定。

## 环境与测量配方

- 硬件/设备：Apple M4 Max；CPU / arm64；1 线程；逻辑 CPU 数 16；物理内存 51539607552 bytes。
- env 指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。
- SUT：478720 参数（约 0.48M）；float32；结构 `{"context_length": 128, "dropout": 0.0, "heads": 4, "layers": 2, "vocab_size": 512, "width": 128}`。
- SUT manifest SHA-256：`8f5f36885a574011e647b4fb046862100f5aa6d4ba99e279d42c0975ac3abfb0`。
- 单进程顺序执行；CLI 在导入 NumPy 前设置各 BLAS/OpenMP 线程上限：`{"BLIS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}`。
- 时延单位 ms；p50/p95/p99 由 measure() 使用 NumPy linear 插值；以下为单轮样本分位数，不是置信区间。
- 预热/重复依据：T4 的全局经验扰动阈值，不是显著性保证。SUT 单次调用已有毫秒级工作量，不以增加 repeat 稀释单样本开销。prefill 预热 100 次、重复 1000 次；decode 每样本包含完整生成回路，预热 10 次、重复 200 次。报告逐项列出 p50/阈值，低于阈值的差异不可分辨。

## 基准数字表

吞吐为推算值：prefill 使用序列 token 数 × 1000 / p50(ms)；decode 使用新增 token 数 × 1000 / 端到端 p50(ms)，不含 prompt token 数。这是 p50 时延倒数推算，非平均吞吐，也不是逐 token 时延分位数。

| 负载 | warmup | repeat | p50 ms | p95 ms | p99 ms | 推算 token/s | peak_tracemalloc_bytes | max_rss_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill-t1 | 100 | 1000 | 0.411167000 | 0.446712200 | 0.493579910 | 2432.101798 | 46860 | 39829504 |
| prefill-t32 | 100 | 1000 | 0.712750000 | 0.817129200 | 0.865986330 | 44896.527534 | 411648 | 50987008 |
| prefill-t128 | 100 | 1000 | 1.567562500 | 1.660994050 | 1.810383670 | 81655.436386 | 1950723 | 62996480 |
| decode | 10 | 200 | 27.083646000 | 29.294422900 | 31.860576580 | 1181.524821 | 614774 | 73728000 |

内存与时延来自同一次 measure()，没有独立计时器。peak_tracemalloc_bytes 是重复测量区间的 traced 峰值，含输出临时分配与样本簿记，不覆盖全部原生分配；模型和固定输入在 tracing 前加载。max_rss_bytes 是 macOS 进程生命周期高水位，包含模型加载及之前负载，不能解释为各负载独立增量或相加。

### 记录与配置哈希

config 哈希来自 measure() 的完整 config（sorted JSON SHA-256），包括配方、负载、SUT 与线程设置；记录路径和采集时间不参与 config 哈希。
每目录包含 config.json、measurement.json（全部样本）、env.lock、meta.json。

| 负载 | config SHA-256 | 采集时间 UTC | runs 目录 |
| --- | --- | --- | --- |
| prefill-t1 | `b39c8fb3a5a4066d7bbc91aae65ad9de9471e3202eb40130eb1e3e27cc742cb7` | 2026-10-19T09:10:33.273005+00:00 | `runs/20261019-baseline-091032830255-prefill-t1` |
| prefill-t32 | `87c4e9140b540dad5dbb094022586cffc5f893ae00c2ad71ffd2963ef8c0812d` | 2026-10-19T09:10:34.048766+00:00 | `runs/20261019-baseline-091032830255-prefill-t32` |
| prefill-t128 | `6770f722303bcfc711ad202907b63b9e0b227b06e0f9a4676df92227ff9e05f6` | 2026-10-19T09:10:35.760355+00:00 | `runs/20261019-baseline-091032830255-prefill-t128` |
| decode | `2bc91ba825084c608d421574f63722a0e8e6737df02927cd4a939e4916f3a670` | 2026-10-19T09:10:41.274524+00:00 | `runs/20261019-baseline-091032830255-decode` |

## T4 分辨率对照

沿用 `results/p0/calibration.md` 的经验阈值 0.092584 ms。它是噪声/扰动尺度，超过阈值仍不能自动证明收益；增加重复次数不能消除 tracing 扰动。

| 负载 | p50 / 阈值 | p95−p50 ms | 判定 | p99−p50 ms | 判定 |
| --- | ---: | ---: | --- | ---: | --- |
| prefill-t1 | 4.441016 | 0.035545200 | 不可分辨 | 0.082412910 | 不可分辨 |
| prefill-t32 | 7.698414 | 0.104379200 | 超过经验阈值，非显著性证明 | 0.153236330 | 超过经验阈值，非显著性证明 |
| prefill-t128 | 16.931246 | 0.093431550 | 超过经验阈值，非显著性证明 | 0.242821170 | 超过经验阈值，非显著性证明 |
| decode | 292.530524 | 2.210776900 | 超过经验阈值，非显著性证明 | 4.776930580 | 超过经验阈值，非显著性证明 |

| prefill p50 对照 | 绝对差 ms | 判定 |
| --- | ---: | --- |
| prefill-t1 / prefill-t32 | 0.301583000 | 超过经验阈值，非显著性证明 |
| prefill-t1 / prefill-t128 | 1.156395500 | 超过经验阈值，非显著性证明 |
| prefill-t32 / prefill-t128 | 0.854812500 | 超过经验阈值，非显著性证明 |

不同序列长度是不同工作量，上表只作分辨率对照，不是优化收益对照。低于经验阈值的差异标为“不可分辨”，不得作为性能收益证据。

## 边界声明

本轮为 CPU/单线程实测，未使用 MPS/CUDA。SUT 无 KV cache，注意力采用 O(n²) 朴素前向；decode 每一步重算完整上下文，超过窗口后按 SUT 规则裁剪。decode 禁用 EOS 提前停止，固定生成配方指定的新 token 数；计时包含 prompt 列表复制、逐步窗口构造、前向与 argmax，不含模型加载、tokenizer 编码和文本解码。prefill 是 batch-one 单次前向，固定输入构造在计时外。这些数字是协议正确性证据，不是生产级性能结论；tracemalloc 扰动、系统调度与温度漂移仍影响观测，单轮结果不支持普遍性能结论。

## 复现与验收入口

- `uv run --locked python scripts/baseline.py`：测量配方、保存新批次记录、覆盖报告。
- `uv run --locked python scripts/baseline.py --check`：读取最新匹配的完整批次并打印，不测量、不写文件；已保存的记录是报告的唯一数字来源。
- `uv run --locked python scripts/baseline.py --from-runs <目录> ...`：从指定完整批次幂等重建报告；加 `--check` 可仅打印。
- 验收：`uv sync --locked`、`uv run --locked pytest`、`uv run --locked ruff check .`、`uv run --locked ruff format --check .`、上述 `--check` 与 `git status --short`。测试中的极小配方仅为 smoke，本报告数字来自正式配方记录。

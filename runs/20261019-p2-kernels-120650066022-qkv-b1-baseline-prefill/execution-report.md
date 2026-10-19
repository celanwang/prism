# P2-T3 执行汇报（任务规格第 6 节）

任务规格：`tasks/p2-t3-kernels.md`；当前分支：`codex/p2-kernels`。
本汇报作为 `-o` 未指定路径时的白名单内持久化副本；最终答复亦提供提交结果。

## 1. 实际改动 / 新增文件

全部为新增文件，未修改或删除原有模块、依赖、锁文件、测量协议或任务规格。
共 106 个文件：5 个实现/测试/配方/报告文件，101 个 run 目录内产物。

- `src/prism/ablating.py`
- `tests/test_ablating.py`
- `configs/p2_kernels.json`
- `scripts/kernel_bench.py`
- `results/p2/kernels.md`

24 个独立 runs 的完整文件清单如下（每项各 8 个，不打包消融）：

- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-decode/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/acceptance-kernel-check.log`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/comparison.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/execution-report.md`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-decode/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b1-qkv_split-prefill/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-decode/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-decode/meta.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill/config.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill/env.lock`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill/measurement.json`
- `runs/20261019-p2-kernels-120650066022-qkv-b2-qkv_split-prefill/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-decode/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-attention_bt-prefill/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-decode/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/comparison.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b1-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-decode/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-attention_bt-prefill/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-decode/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-120910325237-attention-b2-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-decode/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/comparison.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-decode/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b1-mlp_reuse-prefill/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-decode/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-decode/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-decode/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-decode/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-baseline-prefill/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-decode/meta.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill/config.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill/env.lock`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill/measurement.json`
- `runs/20261019-p2-kernels-121132154397-mlp-b2-mlp_reuse-prefill/meta.json`

## 2. 规格第 5 节验收命令输出摘要

| 命令 | 输出摘要 | 退出码 |
| --- | --- | ---: |
| `uv sync --locked` | Resolved 9 packages；Checked 8 packages | 0 |
| `uv run --locked pytest` | 533 passed in 8.39s（新增 44 个用例） | 0 |
| `uv run --locked ruff check .` | All checks passed! | 0 |
| `uv run --locked ruff format --check .` | 60 files already formatted | 0 |
| `uv run --locked prism sut check` | passed=true；greedy_cases_passed=2；逐位置 argmax=32；tokenizer=3；logits 最大绝对误差 9.5367431640625e-06 | 0 |
| `uv run --locked python scripts/kernel_bench.py --check` | 三项各自通过两个 fixture 的 cached/uncached 逐 token 门槛；24 个完整测量臂复跑完成；不新增 save_run 记录 | 0 |
| `git status --short` | 提交前全部新增项均在白名单，暂存范围逐路径审计；提交后工作区状态由最终答复确认 | 0 |

此外：24 个正式记录全部通过 `load_run()`，与验收复跑的 24 个 config hash / env 指纹逐项一致；
源文件 SHA-256 与当前文件一致，p50/p95/p99 从全部原始样本重算一致。
`--check` 的完整 stdout（含原始样本）见同目录 `acceptance-kernel-check.log`。

## 3. 三项消融的对照数据与结论

硬件/设备：Apple M4 Max，CPU / arm64，16 逻辑 CPU，51539607552 bytes 内存；
五个 native 线程环境变量均固定为 1，未使用 MPS/CUDA。
每臂每负载 warmup=100、repeat=2000；p50/p95/p99 使用 NumPy linear 插值。
prefill T=128；decode 为 18-token prompt 后的 32 个 teacher-forced 缓存步，计时不含 prefill。

环境指纹（正式与验收复跑一致，且与 T4 一致）：
`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。

配方文件 sorted JSON SHA-256：
`2fec7e97db4ce6ed43e260435a342fb798d2bd3586d20e7252d151ac49bd93a9`。
每臂实际 measurement.config_hash 不同，完整 24 个映射见 `results/p2/kernels.md` 的追溯表及各 run/config.json。

以下每个数据单元格为 **p50 / p95 / p99（ms）**，decode 是 32 步合计：

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


最终结论纳入正式和完整验收复跑，两次各自按 T4=0.092584 ms 判定，
只保留两次都超过阈值且方向一致的结论；差异低于阈值不作为收益证据。

| 研究项 | prefill 最终结论 | decode 最终结论 | 原因 |
| --- | --- | --- | --- |
| QKV 合并 vs 拆分 | 不可分辨 | 合并值得保留 | 正式拆分多耗 0.097646–0.164146 ms（prefill）、1.530062–1.781729 ms（decode）；复跑 prefill B2 仅 0.086271 ms 低于阈值，decode 四个批次均超过阈值 |
| attention BT 整理 | 不可分辨 | 不值得 | 正式 prefill 退化 0.131125–0.149728 ms，但复跑退化 0.060418 / 0.088979 ms 均低于阈值；decode 正式与复跑都退化且超过阈值 |
| MLP 中间缓冲复用 | 不可分辨 | 不可分辨 | prefill 两次执行均低于阈值；decode 正式一批退化、一批低于阈值且方向改变，验收复跑显示退化，跨执行结论不稳定，未证实时延收益 |

实现与归因限制：attention 仅改变 Q/K/V 临时存储与步长，保留原始 score/softmax；
复制/整理成本计入 forward，缓存 K/V 仍按原规则拥有自有存储。
MLP warm scratch 不包括冷分配成本，traced peak 不包括 tracing 启动前已经分配的 scratch。
T4 是经验扰动参照，不是显著性保证；未估计置信区间，仍有 tracing 与系统漂移的影响。
完整复跑分位数、逐项判定和分析见 [报告草稿](../../results/p2/kernels.md)。

## 4. 与规格的偏差及原因

**无**。三个开关互斥，三项独立配方、独立运行、独立产物；
没有组合优化的测量，没有把低于 T4 的差异解释为收益。
未改依赖、未 git push、未删除文件；负结果和不可分辨结果如实保留。

# P2-T2 执行汇报

任务规格：tasks/p2-t2-spec.md。规划层只读计划已批准；按用户八项裁决执行。当前分支：codex/p2-spec。

## 1. 实际新增文件

新增 INT8 草稿–fp32 块验证回路、115 项测试、k 扫描配方、统一工作负载基准和报告草稿。
生成 38 条正式测量，每条有 config.json、measurement.json、env.lock、meta.json；首条另含 comparison.json、acceptance.json 与本汇报。
本次实际新增 160 个文件；全部位于白名单。没有修改现有文件、依赖、协议模块；未 push、未删除文件。
初轮 115505217673 记录仅本地保留，因报告排版修正导致源码哈希变化，不纳入本次提交或正式证据。

```text
configs/p2_spec.json
results/p2/speculative.md
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/acceptance.json
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/comparison.json
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/execution-report.md
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-fp32-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-fp32-target-token/config.json
runs/20261019-p2-spec-115630579395-b1-fp32-target-token/env.lock
runs/20261019-p2-spec-115630579395-b1-fp32-target-token/measurement.json
runs/20261019-p2-spec-115630579395-b1-fp32-target-token/meta.json
runs/20261019-p2-spec-115630579395-b1-int8-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-int8-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-int8-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-int8-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec1-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b1-spec1-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec1-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b1-spec1-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-spec1-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-spec1-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec1-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-spec1-verify-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec1-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec1-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec1-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec2-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b1-spec2-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec2-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b1-spec2-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-spec2-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-spec2-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec2-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-spec2-verify-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec2-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec2-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec2-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec4-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b1-spec4-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec4-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b1-spec4-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-spec4-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-spec4-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec4-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-spec4-verify-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec4-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec4-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec4-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec8-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b1-spec8-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec8-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b1-spec8-e2e/config.json
runs/20261019-p2-spec-115630579395-b1-spec8-e2e/env.lock
runs/20261019-p2-spec-115630579395-b1-spec8-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec8-e2e/meta.json
runs/20261019-p2-spec-115630579395-b1-spec8-verify-block/config.json
runs/20261019-p2-spec-115630579395-b1-spec8-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b1-spec8-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b1-spec8-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b2-fp32-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-fp32-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-fp32-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-fp32-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-fp32-target-token/config.json
runs/20261019-p2-spec-115630579395-b2-fp32-target-token/env.lock
runs/20261019-p2-spec-115630579395-b2-fp32-target-token/measurement.json
runs/20261019-p2-spec-115630579395-b2-fp32-target-token/meta.json
runs/20261019-p2-spec-115630579395-b2-int8-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-int8-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-int8-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-int8-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec1-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b2-spec1-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec1-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b2-spec1-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-spec1-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-spec1-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec1-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-spec1-verify-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec1-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec1-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec1-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec2-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b2-spec2-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec2-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b2-spec2-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-spec2-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-spec2-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec2-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-spec2-verify-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec2-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec2-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec2-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec4-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b2-spec4-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec4-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b2-spec4-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-spec4-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-spec4-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec4-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-spec4-verify-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec4-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec4-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec4-verify-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec8-draft-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-block/meta.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-sync/config.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-sync/env.lock
runs/20261019-p2-spec-115630579395-b2-spec8-draft-sync/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec8-draft-sync/meta.json
runs/20261019-p2-spec-115630579395-b2-spec8-e2e/config.json
runs/20261019-p2-spec-115630579395-b2-spec8-e2e/env.lock
runs/20261019-p2-spec-115630579395-b2-spec8-e2e/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec8-e2e/meta.json
runs/20261019-p2-spec-115630579395-b2-spec8-verify-block/config.json
runs/20261019-p2-spec-115630579395-b2-spec8-verify-block/env.lock
runs/20261019-p2-spec-115630579395-b2-spec8-verify-block/measurement.json
runs/20261019-p2-spec-115630579395-b2-spec8-verify-block/meta.json
scripts/spec_bench.py
src/prism/speculative.py
tests/test_speculative.py
```

## 2. 规格第 5 节验收摘要

| 命令 | 退出码 | 输出摘要 |
| --- | --- | --- |
| `uv sync --locked` | 0 | Resolved 9 packages in 2ms; Checked 8 packages in 0.16ms |
| `uv run --locked pytest` | 0 | 604 passed in 12.61s; includes 115 new speculative tests |
| `uv run --locked ruff check .` | 0 | All checks passed! |
| `uv run --locked ruff format --check .` | 0 | 61 files already formatted |
| `uv run --locked prism sut check` | 0 | passed=true; greedy cases=2, argmax positions=32, tokenizer prompts=3; max abs logits error=9.5367431640625e-06 |
| `uv run --locked python scripts/spec_bench.py --check` | 0 | {"implementation_passed": true, "hard_gate_cases": 80, "measurements": 38, "writes_runs_results": false} |
| `git status --short` | 0 | Five new whitelisted code/config/report files before force-adding ignored formal runs; final status recorded after commit |

完整验收摘要在同目录 acceptance.json。--check 前后对 runs/ 与 results/ 的 1012 个文件做 SHA256 快照，内容不变；38 条正式记录均经 load_run 校验、每条 30 个 samples、全部源码哈希与最终源码一致。
硬门槛 80 组：fixture 两用例 × k=1/2/4/8；seed=42/20261019 × prompt 长度=1/7/18/64/126/127/128/129/257 × k=1/2/4/8，随机矩阵每例生成 8 tokens。均与 fp32 cached/非 cached 基线逐 token 一致。
逐轮日志独立重建输出、接受率、位置分母和直方图；额外测试覆盖四个拒绝位置、双缓存回滚、EOS、bonus、预算、滑窗、重复调用隔离和失败前不写盘。控制流测试的合成替身仅用于隔离测试，正式草稿保持真实 INT8 权重。

## 3. 接受率–收益曲线和复现锚点

硬件：Apple M4 Max / arm64 / macOS 15.8 / 16 逻辑 CPU / 48 GiB；CPU 单线程。固定 18-token prompt、生成 32 tokens；EOS 关闭，端到端计入 prefill、自回归、日志与同步。
全部测量采用 measure() 原协议：预热 3 次、重复 30 次；端到端两个相反顺序批次，独立组件探针固定顺序。p50/p95/p99 单位 ms，原始 samples 与每条完整 config hash 在 run 中；全表见 results/p2/speculative.md。
环境指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。
配方哈希（sorted JSON）：`329cd7a332914391f4cbbb0e7386744ae3611c17e84c5d716aaa3fb4b60a4253`。
正式 runs：`runs/20261019-p2-spec-115630579395-*`；汇总：`runs/20261019-p2-spec-115630579395-b1-fp32-e2e/comparison.json`。

| batch | arm | 接受率 | 平均投机推进 | p50 | p95 | p99 | 相对 fp32 p50 speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fp32 | — | — | 17.215375 | 17.476300 | 17.559264 | 1.000000 |
| 1 | int8 | — | — | 23.178416 | 25.059117 | 25.233515 | 0.742733 |
| 1 | spec1 | 1.0 | 2.000000 | 37.618750 | 39.308104 | 39.336866 | 0.457628 |
| 1 | spec2 | 1.0 | 2.909091 | 35.769645 | 36.313183 | 36.407513 | 0.481284 |
| 1 | spec4 | 1.0 | 4.571429 | 32.737479 | 33.320673 | 33.689758 | 0.525861 |
| 1 | spec8 | 1.0 | 8.000000 | 29.824917 | 31.117083 | 31.147641 | 0.577215 |
| 2 | spec8 | 1.0 | 8.000000 | 30.543750 | 31.963292 | 33.411626 | 0.550749 |
| 2 | spec4 | 1.0 | 4.571429 | 33.034605 | 33.334759 | 33.594503 | 0.509222 |
| 2 | spec2 | 1.0 | 2.909091 | 35.987437 | 36.496039 | 36.525128 | 0.467439 |
| 2 | spec1 | 1.0 | 2.000000 | 38.715520 | 39.063019 | 39.082093 | 0.434501 |
| 2 | int8 | — | — | 24.584834 | 24.830513 | 24.906899 | 0.684240 |
| 2 | fp32 | — | — | 16.821938 | 16.998228 | 17.000892 | 1.000000 |

全部 k 在两批中相对 fp32 cached、INT8 cached 均无 p50 加速；全部差异绝对值超过 T4=0.092584 ms（results/p0/calibration.md）。超过经验阈值不自动证明显著性。
工作负载接受率均为 1：accepted/proposed 分别为 16/16、22/22、26/26、29/29。平均投机推进分别为 2、2.909091、4.571429、8；终止预算耗尽不出 bonus。
硬门槛全矩阵回退 tokens（k=1/2/4/8）分别为 86/84/84/84，单独计为 fp32 target-only、不进入接受率分母。窗口外收益不外推。
解析模型：独立恒定接受概率 α 下 E[前进]=Σ(i=0..k)α^i，成本 C=kD+V，speedup=T·E/C。D=draft-block mean/k，V=verify-block mean，T=fp32 target-token mean；均在相同真实首轮前缀以 measure() 校准。模型忽略同步、prefill、终止轮及 Python 额外开销，是乐观估算；独立探针不作可加因果归因。

| batch | k | D/T | C/T | α 阈值 | 理想最高 speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.406549 | 2.624198 | 无可行点 | 0.762138 |
| 1 | 2 | 1.415087 | 4.060765 | 无可行点 | 0.738777 |
| 1 | 4 | 1.395090 | 6.845103 | 无可行点 | 0.730449 |
| 1 | 8 | 1.525382 | 13.628452 | 无可行点 | 0.660383 |
| 2 | 1 | 1.500276 | 2.786844 | 无可行点 | 0.717658 |
| 2 | 2 | 1.486622 | 4.268803 | 无可行点 | 0.702773 |
| 2 | 4 | 1.483558 | 7.266327 | 无可行点 | 0.688106 |
| 2 | 8 | 1.482117 | 13.245131 | 无可行点 | 0.679495 |

α=0/0.25/0.5/0.75/0.9/0.95/0.99/1 的完整解析曲线在 speculative.md 与 comparison.json；它们是模型估算，实测观测点为 α=1。D/T 约 1.395–1.525；全部 C/T≥k+1，因此乐观模型在 [0,1] 内也无可行盈亏平衡点。这与端到端负结果一致，不能以接受率=1 推导收益。
同源约 0.5M 小模型使接受率虚高，不可外推到大模型/异源草稿；没有人为改坏正式草稿。INT8 单独精度引用 P2-T1 results/p2/quant.md；所有时延基线均按本次统一工作负载重测。

完整环境锚点：

```json
{
  "environment": {
    "platform": {
      "system": "darwin",
      "machine": "arm64",
      "macos_version": "15.8"
    },
    "python": "3.12.15 (main, Oct  3 2026, 00:54:33) [Clang 22.1.3 ]",
    "cpu_count": 16,
    "memory_bytes": 51539607552,
    "packages": {
      "numpy": "2.5.4",
      "pytest": "9.1.1",
      "ruff": "0.17.1"
    },
    "lock_hash": "80c5d0ad743ae5fc69d61cb4f1398c56481f9a4024da50ce2871cea67f446204",
    "prism_version": "0.1.0",
    "captured_at": "2026-10-19T11:56:30.579028+00:00"
  },
  "runtime": {
    "cpu_model": "Apple M4 Max",
    "device": "cpu",
    "thread_environment": {
      "VECLIB_MAXIMUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "BLIS_NUM_THREADS": "1"
    }
  },
  "anchors": {
    "task": "P2-T2",
    "phase": "P2",
    "sources_sha256": {
      "src/prism/speculative.py": "1faed749de06d2b04203351865c5b158379472718e0325882ee8b94a66b3b11e",
      "scripts/spec_bench.py": "469928ff098c24eec16264ff085f008d0487a844dca3b3c38e7870a6f4fe825a",
      "src/prism/quant.py": "3f3278944ad10e99160c2a31ab68995e5272ab2492acf5fe928a74358d54008a",
      "src/prism/sut.py": "d4cae3336f899e91e28da89550ecc4cd9ddcfea61d4f37c9d5149982947bf517",
      "src/prism/kvcache.py": "88ba181d6b7b714dfcb0f520923dc46c993e54f19bbde558fc65dc9e8aeb988e",
      "src/prism/measure.py": "f40ecfdc1052e1f0fa753db6a76f9a099cf61b02eb535648400539a0f14305f5",
      "src/prism/env.py": "cedd6a413485521a9d05499648627913f54753a64422b348613e8eca82468258",
      "src/prism/record.py": "d30bc47fe1d82879ad84d7ce99f2ba3a55ae154707a03cf54c9ca035125a1e4c"
    },
    "sut_config": {
      "vocab_size": 512,
      "context_length": 128,
      "width": 128,
      "layers": 2,
      "heads": 4,
      "dropout": 0.0
    },
    "manifest_sha256": "8f5f36885a574011e647b4fb046862100f5aa6d4ba99e279d42c0975ac3abfb0",
    "fixture_sha256": "a457353b0412fd06ad4b30eeecaa860623d5fa2b12f85274029ed0f9f3d33bd4"
  }
}
```

## 4. 与规格的偏差

无。八项规划层裁决均按用户授权执行；报告保持草稿状态，待规划层独立验收和终审。
复现：`uv run --locked python scripts/spec_bench.py`；无写盘复核：`uv run --locked python scripts/spec_bench.py --check`。

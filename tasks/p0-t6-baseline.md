# 任务规格 P0-T6：首份基准报告（P0 收尾）

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-baseline`
- 风险等级：低（T1–T5 全部就位后的组装任务；数字解读由规划层负责）
- 依赖：T1 env、T2 SUT、T3 measure、T4 标定、T5 record（均已合并 main）

## 1. 研究问题

P0 的总问题"本机推理的时延/吞吐/内存如何可信测量"的闭环验证：
用 P0 自建的协议对 P0 接入的 SUT 产出第一份有完整证据链的基准报告。
这份报告同时是 P1/P2 一切"收益"的对照基线。

可能的失效形式：

- 测量收益低于测量器分辨率而误判（对策：所有数字附 T4 分辨率阈值对照，
  低于 0.092584 ms 量级的差异标注"不可分辨"）。
- 报告数字无法复现（对策：配方+配置入 configs/，runs/ 原始记录入库，
  报告只从 runs 摘要生成）。

## 2. 改动白名单

- 新增 `scripts/baseline.py`
- 新增 `configs/baseline.json`
- 新增 `tests/test_baseline.py`
- 新增 `results/p0/REPORT.md`
- 新增 `runs/` 下本次基准产生的运行目录（`runs/<日期>-baseline-*/`）
- 修改 `.gitignore`（仅当需要让 runs/ 入库时）

## 3. 禁止事项

- 修改 `src/prism/` 下任何模块（发现缺陷停下汇报，不顺手修）
- 改依赖、`git push`、删除文件、改动测量协议
- 手算或转述数字写入报告——REPORT.md 的数字必须来自 runs/ 记录

## 4. 实现要求

1. `configs/baseline.json` 定义基准配方，至少三类负载
   （参数可调，不写死在代码里）：
   - **prefill**：T2 SUT 单次前向，序列长度取 {1, 32, 128}
   - **decode**：从固定 prompt 贪心生成 32 token 的端到端时延与
     吞吐（token/s，由时延推算并注明推算方式）
   - **内存**：上述负载的 peak_tracemalloc_bytes 与 max_rss_bytes
   每类负载的 warmup/repeat 次数参考 T4 标定结论设定（确保
   被测量显著高于测量器开销），并写入配方。
2. `scripts/baseline.py`：读配方 → 用 `prism.measure.measure()` 测量 →
   用 `prism.record.save_run()` 落盘每个负载的 run → 从 runs 生成
   `results/p0/REPORT.md`（幂等覆盖）。`--check` 模式只打印不写入。
3. `results/p0/REPORT.md` 内容：
   - P0 阶段验收总览：T1–T6 交付物清单与验收证据指针（PR 号、报告位置）
   - 基准数字表：每负载的 p50/p95/p99、吞吐、峰值内存，
     附 env 指纹、config 哈希、runs 目录指针
   - T4 分辨率对照：标注哪些差异低于测量器分辨率
   - 边界声明：CPU/单线程实测、无 KV cache（O(n²) 朴素前向）、
     SUT 规模（~0.5M 参数）；这些数字是协议正确性证据，
     不是生产级性能结论
4. `tests/test_baseline.py`：配方加载校验、负载构造正确
   （序列长度、生成长度符合配方）、REPORT 生成含必需小节；
   用极小配方跑通全流程，不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/baseline.py --check
git status --short
```

人工检查项：REPORT.md 中数字与 runs/ 原始记录一致（规划层抽查两个负载）；
重跑配方后 env 指纹与 config 哈希不变。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 基准数字全表（含 env 指纹、config 哈希）
4. 与规格的任何偏差及原因（无偏差写"无"）

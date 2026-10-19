# 任务规格 P1-T2：分页 KV cache 管理器与内存–吞吐曲线

- 阶段：P1 推理计算（docs/ROADMAP.md）
- 分支：`codex/p1-paged`
- 风险等级：**高**（内存管理数据结构，错误会静默污染后续吞吐实验）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 依赖：P1-T1 已合并（cache 语义与一致性 harness 就位）

## 1. 研究问题

"KV cache 增长如何挤压吞吐"的机制半边：
分页分配（vLLM 式 block table）相对连续分配，在**同一正确性门槛下**
把 cache 内存碎片与峰值占用降低了多少？块大小如何影响
内部碎片与分配开销？驱逐策略（滑窗 vs LRU）在固定内存预算下
对吞吐–命中率的权衡拐点在哪？

可能的失效形式：

- 分页实现与 T1 连续 cache 数值行为不一致（对策：复用 T1 的
  逐 token 一致性 harness，分页路径必须过同一 fixture 门槛）。
- 内存数字来自估算而非实测（对策：以页表实际分配字节数为准，
  解析公式对照，measure() 峰值作旁证）。
- 驱逐正确性未隔离验证（对策：驱逐路径独立的一致性测试，
  含"驱逐后重新需要"的 miss 场景）。

## 2. 改动白名单

- 新增 `src/prism/paged.py`
- 修改 `src/prism/kvcache.py`（仅在分页路径需要抽接口时做最小改动，
  不得改变 T1 连续 cache 路径行为）
- 新增 `tests/test_paged.py`
- 新增 `configs/p1_paged.json`（块大小/驱逐策略扫描配方）
- 新增 `scripts/paged_bench.py`
- 新增 `runs/20261019-p1-paged-*/` 产物目录

## 3. 禁止事项

- 修改 `sut.py`、`measure.py`、`env.py`、`record.py`、`schedsim.py`
- 改依赖、`git push`、删除文件；改动 T1 连续 cache 的数值行为
- 把模拟/推算数字表述为实测

## 4. 实现要求

1. **分页 cache 管理器**：
   - 逻辑 token 块（块大小可配，如 8/16/32）→ 物理块的页表映射；
     物理池总量可配（模拟固定内存预算）；
   - 分配/释放/引用语义明确；fork（共享前缀）本阶段不做；
   - 逐 token 生成时按页表 gather K/V attend，正确性与 T1 连续
     cache 完全一致（同一 fixture 两用例 + 随机 prompt 对照）。
2. **驱逐策略**（物理池耗尽时触发）：
   - 滑窗（逐出最旧逻辑块，与 T1 重建语义对齐）
   - LRU（逐出最久未访问物理块，miss 时重算该块——重算语义
     需与 naive 窗口一致，若无法保证一致则如实标注该策略
     仅用于内存模拟、不参与生成正确性）
3. **内存–吞吐曲线**：`configs/p1_paged.json` 配方扫描
   （块大小 × 物理池预算），每点记录：峰值物理占用、碎片率
   （已分配未使用/已分配）、分配开销（measure() 计时）、
   驱逐次数与命中率；经 save_run() 落盘。
4. 解析模型：分页内存 = 页数 × 块大小 × 每 token 字节数（T1 公式），
   与实测页表占用对照，差异需解释。
5. 风格：ruff line-length 100、全类型注解、docstring 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check
uv run --locked python scripts/paged_bench.py --check
git status --short
```

人工检查项：分页路径逐 token 一致（规划层另起进程抽查）；
碎片率与块大小的单调关系符合预期（块越大内部碎片越高）。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 内存–吞吐曲线数据（含 env 指纹、config 哈希、runs 目录）
4. LRU 驱逐的一致性立场（参与生成 or 仅内存模拟）及理由
5. 与规格的任何偏差及原因（无偏差写"无"）

# 任务规格 P2-T2：投机采样回路（草稿–验证）与接受率–收益曲线

- 阶段：P2 推理优化（docs/ROADMAP.md）
- 分支：`codex/p2-spec`
- 风险等级：**高**（解码正确性门槛微妙：投机采样输出必须与贪心基线逐 token 一致）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 依赖：P1-T1（cached 回路）、P2-T1（量化，草稿模型可用量化权重）

## 1. 研究问题

"投机采样的接受率–收益曲线何时值得"：
用同一模型的**量化版本作草稿、fp32 作验证**（self-speculative），
接受率多高时端到端收益为正？草稿长度 k 与接受率的交互曲线
拐点在哪？在 0.5M 小模型上，草稿–验证分布高度一致，
是否反而使接受率虚高、结论无法外推——如实记录这一边界。

可能的失效形式：

- 投机输出与贪心基线有隐性分歧（对策：逐 token 一致为硬门槛，
  与 P0/P1 同一 fixture）。
- 收益统计口径混淆（对策：分别记录接受率、每步平均前进 token 数、
  端到端 p50/p95/p99 对照，全部 measure() 协议）。
- 草稿模型质量未控制变量（对策：草稿=INT8 量化版，其单独精度
  数据引 P2-T1 结果）。

## 2. 改动白名单

- 新增 `src/prism/speculative.py`
- 新增 `tests/test_speculative.py`
- 新增 `configs/p2_spec.json`（k 扫描配方）
- 新增 `scripts/spec_bench.py`
- 新增 `runs/<日期>-p2-spec-*/` 产物目录
- 新增 `results/p2/speculative.md`（报告草稿，规划层终审）

## 3. 禁止事项

- 修改 `sut.py`、`kvcache.py`、`quant.py`、`measure.py`、`env.py`、`record.py`
- 改依赖、`git push`、删除文件
- 放宽逐 token 一致门槛；把单点收益外推为普遍结论

## 4. 实现要求

1. **投机解码回路**：草稿模型（INT8 量化，复用 P2-T1）自回归
   生成 k 个候选 → 验证模型（fp32 + T1 cached）一次前向并行打分
   → 按经典算法逐位接受/拒绝并重采样该位（贪心下即取 argmax
   比对，拒绝时以验证模型 argmax 续位）；
   **输出序列必须与 fp32 贪心基线逐 token 一致**。
2. **接受率统计**：每步接受的草稿 token 数分布、平均前进长度、
   按位置的接受率衰减；k 扫描（如 1/2/4/8）。
3. **收益对照**：`configs/p2_spec.json` 配方测端到端 decode
   p50/p95/p99（对照 fp32 cached 基线与 P2-T1 量化基线），
   附 T4 阈值判定；盈亏平衡点（收益为正的最低接受率）给出
   解析推算并与实测对照。
4. 报告草稿：接受率–收益曲线、小模型边界声明
   （草稿–验证同源导致接受率虚高，不可外推到大模型场景）。
5. 风格：ruff line-length 100、全类型注解、docstring 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check
uv run --locked python scripts/spec_bench.py --check
git status --short
```

人工检查项：投机输出与贪心基线逐 token 一致（规划层另起进程抽查
随机 prompt）；接受率统计与逐位日志一致。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 接受率–收益曲线数据（含 env 指纹、config 哈希、runs 目录）
4. 与规格的任何偏差及原因（无偏差写"无"）

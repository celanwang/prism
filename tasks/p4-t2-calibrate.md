# 任务规格 P4-T2：训练系统单机小规模实测校准与 P4 报告

- 阶段：P4 训练系统（模拟为主，docs/ROADMAP.md）
- 分支：`codex/p4-calibrate`
- 风险等级：低
- 依赖：P4-T1 已合并；kaleido 训练回路作只读参考

## 1. 研究问题

P4 的收口问题：P4-T1 的解析模型在**能真实测量的最小规模**上
偏差多大？本机可测的范围是：单进程 DP=1 的显存/步时
（训练步的内存与时间）、微批大小对激活显存的线性性。
解析模型哪些假设在此规模已可见误差，哪些必须留待多卡验证？

可能的失效形式：

- 用推理回路冒充训练测量（对策：被测件是真实训练步——
  前向+反向+优化器更新，可用 kaleido 训练回路只读参考，
  prism 侧用 NumPy 实现最小训练步或复用 kaleido 导出的
  训练记录作对照，规格执行时二选一并说明）。
- 单点校准冒充模型验证（对策：micro-batch 取 {1,2,4} 三点，
  检验激活显存的线性预测）。

## 2. 改动白名单

- 新增 `scripts/p4_calibrate.py`
- 新增 `configs/p4_calibrate.json`
- 新增 `tests/test_p4_calibrate.py`
- 新增 `results/p4/REPORT.md`
- 新增 `runs/<日期>-p4-calibrate-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 现有模块；改依赖、`git push`、删除文件
- 把单机实测外推为多卡结论；修改 P4-T1 公式（发现错误停下汇报）

## 4. 实现要求

1. **最小训练步测量**：对 prism SUT 规模的模型实现/复用
   真实训练步（forward+backward+SGD/AdamW 一步），
   micro-batch ∈ {1,2,4}，measure() 测步时与峰值内存，
   save_run() 落盘。
2. **校准对照**：P4-T1 解析模型在同参数下的预测 vs 实测，
   逐项（参数/梯度/优化器/激活）报告绝对与相对偏差；
   激活随 micro-batch 的线性性检验。
3. **results/p4/REPORT.md**：P4 阶段报告——解析模型要点、
   校准数据、偏差分析（哪些假设主导误差）、明确边界
   （单机 CPU、无多卡、通信量未实测仅为解析值）。
4. 测试：流程可跑（极小配置）、偏差计算正确、报告含必需小节。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/p4_calibrate.py --check
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 校准数据全表与偏差分析
4. 与规格的任何偏差及原因（无偏差写"无"）

# 任务规格 P1-T4：模拟–实测互校准与 P1 阶段报告

- 阶段：P1 推理计算（docs/ROADMAP.md）
- 分支：`codex/p1-report`
- 风险等级：低（组装与测量任务；结论解读由规划层负责）
- 依赖：P1-T1（cache 回路）、P1-T2（分页）、P1-T3（模拟器）均已合并

## 1. 研究问题

P1 的收口问题：T3 模拟器里的服务时间模型（prefill 斜率、
decode 随批大小的分段线性节点、0.35 批伸缩假设）与真实回路差多少？
校准后，模拟器对"静态批 vs continuous batching"的定性结论是否改变？
这是 kaleido 方法论的核心：模拟必须经实测校准才算证据。

可能的失效形式：

- 校准点太少导致过拟合（对策：批大小取 {1,2,4,8} 四点实测，
  模拟器在同一批大小上对照，报告逐点偏差）。
- 真实批处理不存在（SUT 是 batch-1 回路），硬造并发（对策：
  实测"同一进程内串行执行 B 个序列的一步 decode"作为批成本
  代理，如实标注代理性质与局限）。

## 2. 改动白名单

- 新增 `scripts/p1_calibrate.py`
- 新增 `configs/p1_calibrate.json`
- 新增 `tests/test_p1_calibrate.py`
- 新增 `results/p1/REPORT.md`
- 新增 `runs/<日期>-p1-calibrate-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 下模块（发现缺陷停下汇报）
- 改依赖、`git push`、删除文件
- 把校准后的模拟结论表述为实测结论

## 4. 实现要求

1. **批成本实测代理**：用 T1 cached 回路，对批大小 B∈{1,2,4,8}：
   同一进程内构造 B 个独立序列（各自 cache），测"每个序列
   各前进一步"的串行总耗时 ÷ 1 步，作为 decode(B) 实测代理；
   prefill 斜率用 T∈{1,32,128} 三点拟合。全部经 measure()、
   save_run() 落盘，warmup/repeat 参照 T4 设定。
2. **模拟器校准**：用实测节点替换 T3 配置中的解析外推
   （生成校准后的配置副本，不改原 p1_schedsim.json），
   重跑代表性网格点（λ=100/s、B=4 两种策略），
   输出校准前后对照。
3. **偏差分析**：逐点报告 实测 vs 模拟服务时间 的相对偏差；
   定性结论（continuous 优于静态批、雪崩点位置）校准前后
   是否改变，如实陈述。
4. **results/p1/REPORT.md**：P1 阶段验收报告——
   T1–T4 交付物与证据指针（PR 号）、KV cache 正确性门槛结果、
   加速比与内存数字（引 T1 runs）、分页内存–吞吐曲线（引 T2 runs）、
   模拟–实测校准结论、边界声明（CPU、batch-1 代理、小模型）。
5. 测试：校准脚本流程可跑（极小配方）、报告含必需小节、
   偏差计算正确；不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/p1_calibrate.py --check
git status --short
```

人工检查项：REPORT 数字与 runs 记录一致（规划层抽查两点）；
校准前后模拟对照可复现。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 校准数据全表与偏差分析摘要
4. 与规格的任何偏差及原因（无偏差写"无"）

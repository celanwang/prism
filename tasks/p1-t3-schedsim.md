# 任务规格 P1-T3：continuous batching 调度模拟器

- 阶段：P1 推理计算（docs/ROADMAP.md）
- 分支：`codex/p1-schedsim`
- 风险等级：低（独立纯模拟模块，与 P1-T1/T2 无代码依赖，可并行）
- 依赖：T3 measure（统计口径）、T5 record（结果落盘）

## 1. 研究问题

"批大小与调度策略的权衡拐点在哪"的模拟半边：
在**不需要真实 GPU 并发**的前提下，continuous batching 相比静态批处理，
对吞吐与尾时延的影响随负载（到达率、序列长度分布）如何变化？
队列多长开始雪崩？

可能的失效形式：

- 模拟器时间模型拍脑袋，与真实回路脱节（对策：服务时间分布参数
  来自 P0/T1 实测分布的解析近似，参数入配置并注明来源；
  互校准在 P1-T4 用真实小规模回路验证）。
- 只报均值吞吐，尾时延与雪崩点被掩盖（对策：p50/p95/p99 时延、
  队长时间序列、拒绝/超时计数全量输出）。

## 2. 改动白名单

- 新增 `src/prism/schedsim.py`
- 新增 `tests/test_schedsim.py`
- 新增 `configs/p1_schedsim.json`（扫描配方）
- 新增 `scripts/schedsim.py`（跑扫描、落盘 runs）

## 3. 禁止事项

- 修改 `src/prism/` 现有模块（sut/measure/env/record/kvcache 只 import）
- 改依赖、`git push`、删除文件
- 把模拟结论表述为实测结论（输出与文档中必须标注"模拟"）

## 4. 实现要求

1. **事件驱动模拟器**（离散事件，纯 Python+NumPy，单进程确定性）：
   - 请求按泊松到达（率 λ 可配），prompt 长度与输出长度服从
     可配分布（均匀/截断正态，参数入配置）；
   - 服务模型：decode 每步耗时 = f(批大小)，f 为分段线性函数，
     节点值入配置（来源注明：P0/T1 实测的解析外推，P1-T4 校准）；
     prefill 一次性耗时按 token 数线性；
   - 调度策略至少两种：静态批（攒满 B 或超时 t 发车）、
     continuous batching（每步可插入新请求，批上限 B、max batched tokens 可配）；
   - 全局随机种子入配置，同配置两次运行结果逐事件一致。
2. **输出**：每请求等待/端到端时延样本、p50/p95/p99、吞吐（完成 token/s）、
   最大队长、雪崩判定（稳态队长随时间无界增长或 p99 超过可配阈值）；
   结果经 `save_run()` 落盘，measurement 字段放统计摘要，
   完整样本存 runs 目录附加文件。
3. **扫描**：`scripts/schedsim.py` 按 `configs/p1_schedsim.json` 扫
   λ × B 网格，输出汇总表（标注模拟）。
4. **测试**：确定性（同种子两次逐事件一致）、Little's law  sanity
   检查（低负载下平均队长 ≈ λ×平均时延，容差内）、两种策略在
   极低负载下行为一致、雪崩场景能被判定；不断言绝对性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/schedsim.py --check   # 小规模自检，不写入
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 一个代表性网格点的两种策略对照数字（标注模拟、附 config 哈希）
4. 与规格的任何偏差及原因（无偏差写"无"）

# 任务规格 P3-T2：队列/背压压测与 P3 阶段报告

- 阶段：P3 服务与部署（docs/ROADMAP.md）
- 分支：`codex/p3-report`
- 风险等级：低
- 依赖：P3-T1 已合并

## 1. 研究问题

"队列多长开始雪崩"的实测半边：
对 P3-T1 的 HTTP 服务做负载扫描（到达率 × 并发度），
找到 p99 尾时延拐点与 429 背压触发点；
实测雪崩行为与 P1-T3 模拟器（经 P1-T4 校准）的预测差多少？

可能的失效形式：

- 压测压垮的是客户端而非服务（对策：客户端连接池与超时
  入配置，客户端开销单独记录）。
- 雪崩点判定主观（对策：判定规则复用 T3 的 p99 阈值 +
  队长增长代理，写入配方）。

## 2. 改动白名单

- 新增 `scripts/serve_load.py`
- 新增 `configs/p3_load.json`
- 新增 `tests/test_serve_load.py`
- 新增 `results/p3/REPORT.md`
- 新增 `runs/<日期>-p3-load-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 模块；改依赖、`git push`、删除文件
- 把单进程玩具服务的压测数字表述为生产级结论

## 4. 实现要求

1. **负载扫描**：`configs/p3_load.json` 网格——到达率 ×
   并发客户端数 × 队列上限；固定种子请求序列；
   记录每点 p50/p95/p99、429/超时计数、吞吐。
2. **雪崩判定**：复用 T3 规则（p99 阈值 + 队长增长代理），
   输出实测雪崩点；与校准后模拟器同参数对照，报告偏差。
3. **results/p3/REPORT.md**：P3 阶段报告——部署等价性证据
   （引 T1）、并发–尾时延曲线、背压行为、雪崩点实测 vs 模拟、
   边界声明（单进程串行推理、loopback、无 TLS）。
4. 测试：负载生成确定性、判定逻辑、报告小节；不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/serve_load.py --check
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 压测曲线与雪崩点数据（含 env 指纹、config 哈希）
4. 与规格的任何偏差及原因（无偏差写"无"）

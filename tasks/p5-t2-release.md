# 任务规格 P5-T2：独立进程复跑与发布候选审阅（P5 收口）

- 阶段：P5 复现与发布（docs/ROADMAP.md）
- 分支：`codex/p5-release`
- 风险等级：低（验证与文档任务，但验收标准最高）
- 依赖：P5-T1 已合并

## 1. 研究问题

项目总收口：一个不知道开发过程的独立进程，能否仅凭仓库
内容验证全部证据链？发布候选（v0.1.0）的完整性、
文档与证据的一致性是否达标？

可能的失效形式：

- 复跑在"原污染环境"里假绿（对策：干净 clone/worktree +
  新 venv + 独立进程，全流程录证）。
- 文档与现状漂移（对策：README/DESIGN/ROADMAP/AGENTS 与
  代码实际行为逐条核对，差异清单入报告）。

## 2. 改动白名单

- 新增 `scripts/release_check.py`
- 新增 `configs/release.json`
- 新增 `tests/test_release.py`
- 新增 `results/p5/REPORT.md`
- 修改 `README.md`（仅补齐复现与验证指引）
- 修改 `docs/ROADMAP.md`（仅阶段状态与证据指针更新）
- 新增 `runs/<日期>-p5-release-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 模块；改依赖、`git push`、删除文件
- 为通过验收而修改任何历史 runs/ 记录

## 4. 实现要求

1. **独立复跑**：`scripts/release_check.py` 在临时目录
   clone 仓库 → `uv sync --locked` → 全量测试 →
   `prism verify` → 按 P5-T1 的复跑抽样执行代表性配方 →
   与入库记录比对（config 哈希必等、env 指纹差异报告、
   性能数字按 T4 阈值标注）；全程日志落盘。
2. **文档一致性核对**：README 的每条命令实测可跑；
   ROADMAP 阶段状态与 results/ 证据一一对应；
   AGENTS.md 约定与实际流程一致。差异清单写入报告。
3. **results/p5/REPORT.md**：发布候选审阅——证据链总览
   （P0–P4 报告与 PR 指针）、独立复跑结果、文档核对结论、
   已知边界与免责声明汇总、版本建议（v0.1.0 或暂缓）。
4. 测试：release_check 流程可跑（--check 小规模）、
   清单解析、比对逻辑；不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/release_check.py --check
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 独立复跑与文档核对结果
4. 与规格的任何偏差及原因（无偏差写"无"）

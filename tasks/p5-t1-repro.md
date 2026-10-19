# 任务规格 P5-T1：统一 CLI 与全仓复现入口

- 阶段：P5 复现与发布（docs/ROADMAP.md）
- 分支：`codex/p5-repro`
- 风险等级：**高**（触及全部阶段的入口与工件完整性）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 依赖：P0–P4 全部合并

## 1. 研究问题

"prism 的全部实验证据能否由一条命令在干净环境中重建"：
统一 CLI 把各阶段的配方串成可复现流水线，工件完整性
（哈希清单）可机器校验，独立进程复跑结果与入库记录一致。

可能的失效形式：

- 复现入口依赖开发者机器状态（对策：干净 worktree + 新进程
  复跑验证；env 指纹不一致时明确报告而非静默通过）。
- 完整性校验漏掉生成物（对策：工件清单覆盖 data/runs/results
  的关键文件，哈希来自入库版本）。

## 2. 改动白名单

- 修改 `src/prism/cli.py`
- 新增 `src/prism/reproduce.py`
- 新增 `tests/test_reproduce.py`
- 新增 `configs/reproduce.json`
- 新增 `artifacts/inventory-v1.json`（工件哈希清单）
- 新增 `scripts/verify_inventory.py`

## 3. 禁止事项

- 修改 `src/prism/` 其他现有模块；改依赖、`git push`、删除文件
- 复跑时修改任何 runs/ 入库记录（复跑产物写新目录）

## 4. 实现要求

1. **统一 CLI 子命令**（挂 `prism` 入口）：
   - `prism reproduce --list`：列出全部可复现配方及依赖关系
   - `prism reproduce run <recipe>`：执行指定配方
   - `prism reproduce --all`：按依赖顺序执行全部（支持 --dry-run）
   - `prism verify`：按 inventory 校验工件哈希，不符即非零退出
2. **工件清单**：`artifacts/inventory-v1.json` 覆盖 data/sut 工件、
   各阶段关键 runs 记录与 results 报告，逐文件 sha256；
   `scripts/verify_inventory.py` 独立校验。
3. **复跑一致性**：`configs/reproduce.json` 定义复跑抽样
   （每阶段至少一个代表性配方），复跑结果的 config 哈希、
   env 指纹与入库记录比对，差异如实报告（性能数字允许
   漂移但需引用 T4 阈值标注）。
4. 测试：清单加载与校验（含篡改检出）、CLI 子命令行为、
   复现依赖拓扑排序正确；不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism verify
uv run --locked prism reproduce --all --dry-run
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 复跑抽样的一致性结果
4. 与规格的任何偏差及原因（无偏差写"无"）

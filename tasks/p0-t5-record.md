# 任务规格 P0-T5：实验记录规范（runs/ schema 与报告生成）

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-record`
- 风险等级：低
- 依赖：T3 已合并；与 T4 无代码冲突（不同文件），可并行

## 1. 研究问题

实验证据要可复现，记录格式必须先于实验存在。本任务回答：
一次测量运行（run）以什么目录结构与元数据落盘，
才能让别人（和未来的自己）从原始记录重建结论？

可能的失效形式：

- 手改 results/ 导致证据不可信（对策：results 只由工具从 runs 生成，
  CI/测试检查生成一致性）。
- 记录缺 env 指纹或 config，无法判断两次运行是否可比（对策：schema
  强制字段，缺失即拒绝生成报告）。

## 2. 改动白名单

- 新增 `src/prism/record.py`
- 新增 `tests/test_record.py`
- 修改 `src/prism/cli.py`（挂接 `prism record` 子命令）
- 新增 `docs/RUNS.md`（目录规范说明，一页以内）

## 3. 禁止事项

- 修改 `measure.py`、`env.py`、`sut.py`；改依赖；`git push`；删除文件
- 实现任何实际测量业务逻辑（本任务只管记录格式与生成）

## 4. 实现要求

1. **runs/ 目录 schema**：一次运行一个目录 `runs/<YYYYMMDD>-<slug>/`，内含：
   - `config.json`：运行配置（任意 dict，落盘即定哈希）
   - `measurement.json`：T3 Measurement 的 to_json 输出
   - `env.lock`：运行时 collect_env() 快照
   - `meta.json`：创建时间、prism 版本、关联任务/阶段标签
2. `src/prism/record.py` 提供：
   - `save_run(runs_dir, slug, config, measurement) -> Path`：
     原子写入（先写临时目录再 rename），拒绝覆盖已存在目录
   - `load_run(path) -> Run`：校验四件齐全、config 哈希与
     measurement.config_hash 一致、env 指纹可计算，缺项/不符抛错
   - `summarize(runs_dir) -> str`：汇总所有合法 run 的摘要表
     （slug、p50/p95/p99、峰值内存、env 指纹、config 哈希前 8 位）
3. CLI：`prism record save --slug <s> --config <json文件>`（读 stdin 或
   文件的 measurement）、`prism record summary`。
4. `docs/RUNS.md` 写清：目录结构、字段语义、"results/ 只从 runs/ 生成、
   禁止手改"的纪律。
5. 测试：save/load 往返、重复 slug 拒绝、篡改 config 后 load 抛错、
   缺文件抛错、summary 输出包含全部 run。不断言性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism record summary   # 空 runs/ 时应给出空表而非报错
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 与规格的任何偏差及原因（无偏差写"无"）

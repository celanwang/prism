# Prism

棱镜：以可复现实验理解 AI 基础设施的设计、取舍与失效边界。

原始愿景见 [DESIGN](DESIGN.md)，阶段与验收见 [ROADMAP](docs/ROADMAP.md)，
双 agent（Kimi 规划把关 + Codex 执行）协作约定见 [AGENTS](AGENTS.md)。

当前处于立项骨架阶段，P0（测量地基）尚未开始。

## 运行

环境：Python 3.12，uv，Apple Silicon Mac（CPU/MPS）；无 GPU/CUDA 依赖。

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m prism
```

## 目录

- `src/prism/` — 实现代码
- `tests/` — 测试
- `configs/` — 实验配方配置
- `results/` — 阶段验收报告与正式证据摘要
- `runs/` — 单次运行产物（不入库）
- `docs/` — ROADMAP 与设计文档

## 复现与验证（v0.1.0 仓库候选）

以上“P0 尚未开始”为立项时的历史说明；当前 P0–P4 已验收合并，
P5 的候选审阅和独立复跑见 [发布审阅报告](results/p5/REPORT.md)。
请在完整仓库 clone 根目录执行上面的环境与测试命令，然后执行：

```sh
uv run --locked prism verify
uv run --locked python scripts/verify_inventory.py
uv run --locked prism reproduce --list
uv run --locked prism reproduce --all --dry-run
uv run --locked prism reproduce --sample
uv run --locked python scripts/release_check.py --check
```

`verify` 按冻结的 inventory 检查历史工件，缺失或篡改均返回非零。
`--list` 展示配方与依赖，`--all --dry-run` 仅列全量计划；
`--sample` 执行 P0–P4 的完整代表配方（包含依赖），会实际测量，不是 smoke。
复跑的日志、原始记录与比对 JSON 在新建的 `runs/.reproduce/` 会话中。
配置哈希必须相等；环境指纹差异单列报告；p50/p95/p99 漂移按 P0-T4
0.092584 ms 经验扰动尺度标注，不设速度合格线，也不代表统计显著性。
新拟合的模拟结果不冒充历史相同，模拟与实测边界详见配置和阶段报告。

发布脚本的 `--check` 是小规模 clone、新 venv、CLI 校验流程；
去掉该参数才执行全量测试、README 全部 shell 命令和完整抽样的正式发布审阅。
每次新建 `runs/<UTC日期>-p5-release-*/`，其中有候选快照补丁与哈希、逐进程日志、
比对 JSON、原始复跑记录归档；未提交候选只纳入 P5-T2 白名单源文件，
先在临时仓库中提交快照，再从快照进行干净 clone，不带入开发 venv。
独立进程仍在同一硬件上运行，不等于跨硬件或第三方独立验收。

发布范围为仓库内 CLI；wheel 不包含 scripts/configs/inventory/SUT，
不能单独完成复现。现有实验限 CPU；MPS/CUDA/多卡未实测。
`runs/` 默认忽略新产物，但已有部分正式历史证据明确入库并受 inventory 保护；
P5 只新增发布目录，不覆盖历史记录。阶段状态以 ROADMAP 和 results/ 报告为准。

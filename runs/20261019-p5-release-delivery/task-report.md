# P5-T2 执行汇报（任务规格第 6 节）

任务规格：`tasks/p5-t2-release.md`。当前分支：`codex/p5-release`。
已先 fetch 并 rebase 到包含 P5-T1 PR #17 的 origin/main（`90ce36d`）；
任务执行基线为 `3a72da1e7b609359c568e4d07d117f4f7091bfac`。

## 1. 实际改动/新增文件

- 新增 `scripts/release_check.py`：临时候选快照、干净 clone、新 venv、独立进程执行、全程日志、比对、文档清单、归档及哈希；默认完整抽样，`--check` 明确为小规模 smoke。
- 新增 `configs/release.json`：版本候选、超时、P0–P5 报告及合并 PR 清单；复用 P5-T1 配方与 T4 阈值。
- 新增 `tests/test_release.py`：10 项测试，包含真实 clone、新 venv 流程、解析拒绝、缺失证据、比对负例、环境差异、失败留证、归档历史排除；不断言机器性能。
- 新增 `results/p5/REPORT.md`：证据链总览、两次正式复跑、文档差异、边界、版本建议与最终验收。
- 修改 `README.md`：仅追加复现/验证指引及旧状态说明的解释。
- 修改 `docs/ROADMAP.md`：仅更新 P5 状态与证据指针；候选审阅完成，待规划层独立验收与合并。
- 新增 `runs/20261019-p5-release-125334130717-full/`：首次路径拒绝的失败记录，保留不删除。
- 新增 `runs/20261019-p5-release-125418964850-full/`：首轮正式复跑，原始记录、日志、候选补丁与比对。
- 新增 `runs/20261019-p5-release-125918660239-full/`：最终脚本正式复跑，同上并含证据哈希。
- 新增 `runs/20261019-p5-release-130339649116-check/`：最终验收的小规模流程记录。
- 新增 `runs/20261019-p5-release-delivery/`：本汇报、验收日志、原始归档重载复核及源文件一致性复核。

逐文件完整清单见 [changed-files.txt](changed-files.txt)，包含所有 candidate 副本、补丁、逐进程日志与压缩归档。
没有修改 src/prism、依赖/lock、历史 runs 或 P0–P4 报告，没有删除仓库文件或 push。

## 2. 每条验收命令输出摘要

| 命令 | 实际输出摘要 | 返回码与日志 |
| --- | --- | --- |
| `uv sync --locked` | Resolved 9 packages；Checked 8 packages | 0，[01-sync.log](01-sync.log) |
| `uv run --locked pytest` | 751 passed in 37.79s | 0，[02-pytest.log](02-pytest.log) |
| `uv run --locked ruff check .` | All checks passed! | 0，[03-ruff-check.log](03-ruff-check.log) |
| `uv run --locked ruff format --check .` | 87 files already formatted | 0，[04-ruff-format.log](04-ruff-format.log) |
| `uv run --locked python scripts/release_check.py --check` | ok=true，smoke_only=true；新 clone、新 venv、CLI 测试、1591文件 verify、SUT 复跑通过 | 0，[05-release-check.log](05-release-check.log) |
| `git status --short` | 提交前只有六个白名单源文件项；runs 按仓库约定忽略，提交时显式纳入证据 | 0，[06-git-status.log](06-git-status.log) |

机器可读状态和各命令耗时见 [acceptance.json](acceptance.json)。
此外六个白名单源文件的 `git diff --cached --check -- <源文件清单>` 通过；
完整暂存区的额外空白检查会标注 candidate.patch 中合法的空行上下文前缀空格，原始补丁及哈希保持不变。
最终发布代码/测试/配置/README/ROADMAP 与正式复跑输入哈希完全相等，
本报告和 results/p5/REPORT.md 为测量后的生成总结，见 [provenance-validation.json](provenance-validation.json)。
提交格式：`feat(release): audit clean-clone replay and v0.1.0 candidate`。

## 3. 独立复跑与文档核对结果

正式模式实际执行了临时 seed clone → 白名单候选提交 → 第二次干净 clone →
`uv sync --locked` → 全量测试 → `prism verify` → README 的全部11条 shell 命令 →
完整 P5-T1 sample 与依赖 → 状态复核。uv 下载缓存允许复用，不继承原 venv 或 Python/uv/git 覆盖变量。
最终独立 clone 的全量测试为 751 passed（34.76s），两条 ruff 通过，1591个冻结文件校验通过，结束 status 为空。

两轮均为10配方、93个标准 run、67次历史锚点配置哈希相等、环境差异0、5次冻结模拟/解析产物一致；
16个 unanchored 网格点和10个 new-derived 模拟结果没有冒充历史一致。
T4 没有标准 run，只比对报告配置/环境指纹。相同历史锚点在不同配方下的比对次数单独计数。
最终轮 p50/p95/p99 有34/41/45次漂移超过冻结的0.092584 ms经验尺度，无速度合格线、无显著性宣称。

性能边界：Apple M4 Max CPU、arm64、16逻辑CPU、48GiB、macOS15.8、Python3.12.15、NumPy2.5.4，原生线程上限1；
完整预热/重复、分位数定义、测量 config hash 与逐项 signed drift 见 results/p5/REPORT.md 及原始归档。
抽样配置字节 SHA-256 为 `055ca68a289463af9914b9dd5532c0ba38bc84befaad88deca7025eef9920ad0`，
环境指纹为 `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。

原始归档分别解包重载93个 run 后，重算比对结果与摘要完全相等；最终69个证据哈希再次核验相等，
见 [artifact-validation.json](artifact-validation.json)。最终原始归档 SHA-256：
`7db1d9232ef22671a44c9e670616cba86c5f619722e3ffa6f1da0204a35fbb0d`。

文档核对：README 11条命令均实际通过；ROADMAP P0–P4 与报告、inventory、合并PR对应；
P2 使用三个分项报告；P5 单独列本分支待规划层验收。
DESIGN 的CPU/模拟边界一致，AGENTS 的规划/执行/验收职责保持不变。
历史 README 旧状态、P0/P1 无PR措辞、P2 草稿措辞、P1 KV量化缺项、wheel 不含复现资源、
P5 新证据不在旧 inventory 等差异及边界全部列入 P5 报告，没有改历史记录来掩盖差异。

版本建议：v0.1.0 作为完整仓库 clone 的研究 CLI 候选。不是独立 wheel、跨硬件复现、生产容量保证或多卡实测。
规划层仍须按 AGENTS 独立验收并决定合并/发布；本次未发布。

## 4. 与规格的偏差

**无。** 首次正式尝试因 macOS `/var` 软链接而被路径检查拒绝，已修复临时路径规范化并保留失败记录。
随后正式复跑与最终验收均通过。该失败是实现过程中如实记录的负结果，不是放宽规格或测量协议。

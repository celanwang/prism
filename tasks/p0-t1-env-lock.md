# 任务规格 P0-T1：环境与工具链锁定

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-env-lock`
- 风险等级：低（规格 → 执行 → 验收，三步流程）

## 1. 研究问题

"本机推理的时延/吞吐/内存如何可信测量"的前提是测量环境可陈述、可复现。
本任务回答：prism 如何以一条命令给出当前实验环境的完整、机器可读描述，
作为后续所有实验记录的 env 指纹。

可能的失效形式：

- 环境信息缺项（如无依赖哈希），导致两次"相同配置"的实验实际环境不同而无法发现。
- 输出非结构化或包含非确定性字段未加隔离，导致 env 指纹无法做相等性比较。

## 2. 改动白名单

允许修改/新增以下路径，其余一律禁止：

- 新增 `src/prism/env.py`
- 修改 `src/prism/cli.py`（仅接入新子命令）
- 新增 `tests/test_env.py`
- 修改 `tests/test_cli.py`（仅当现有断言与新 CLI 输出冲突时做最小适配）
- 新增 `configs/env.lock`（由命令生成的基线指纹，提交入库）

## 3. 禁止事项

- 修改 `pyproject.toml`、`uv.lock`；引入任何新第三方依赖（只允许标准库 + 已有 numpy）
- `git push`、操作任何远端
- 删除任何现有文件
- 触碰 kaleido 仓库或 prism 以外的任何路径
- 改动测量协议相关设计（本任务不含测量逻辑）

## 4. 实现要求

1. `src/prism/env.py` 提供 `collect_env() -> dict`，采集至少以下字段：
   - `platform`：系统（darwin）、machine（如 arm64）、macOS 版本
   - `python`：完整版本串
   - `cpu_count`：逻辑核数
   - `memory_bytes`：物理内存总量（macOS 用 `sysctl -n hw.memsize`，
     只允许标准库 subprocess 调用）
   - `packages`：numpy / pytest / ruff 的实际安装版本（用 importlib.metadata）
   - `lock_hash`：`uv.lock` 文件内容的 sha256（相对仓库根解析，找不到时置 null）
   - `prism_version`
   - `captured_at`：ISO 8601 时间戳。注意：该字段是唯一非确定字段，
     指纹比较语义上必须可被排除——提供 `env_fingerprint(env: dict) -> str`，
     定义为"剔除 `captured_at` 后按 key 排序的 JSON 的 sha256"。
2. CLI 接入子命令：
   - `prism env`：向 stdout 打印 `collect_env()` 的 JSON（`sort_keys=True`，缩进 2）
   - `prism env --write`：写入 `configs/env.lock`（同一份 JSON，末尾换行），
     并向 stdout 打印写入路径与指纹
   - 不带子命令时的原占位输出可以保留，也可替换为帮助信息（二选一，不要两者混排）
3. 代码风格遵守仓库既有约定（ruff line-length 100，全类型注解，
   docstring 风格与现有文件一致）。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism env            # 输出须为合法 JSON 且含上述全部字段
uv run --locked prism env --write    # 生成 configs/env.lock
git diff --stat                      # 改动不得越出白名单
```

另有人工检查项：`env_fingerprint` 对同一 dict 两次调用结果一致；
剔除时间戳后重跑 `prism env` 两次指纹一致。

## 6. 汇报要求

完成后在最终汇报（`-o` 输出文件）中列出：

1. 实际改动/新增的文件清单（逐文件）
2. 上述每条验收命令的实际输出摘要
3. 与规格的任何偏差及原因（无偏差写"无"）

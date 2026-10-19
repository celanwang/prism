# 任务规格 P0-T2：被测件接入（kaleido P3 参考模型）

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-sut`
- 风险等级：**高**（跨仓库工件复用 + 后续所有基准的被测对象）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 版本：v2（2026-10-19）。v1 经只读计划预审发现两处缺陷后由规划层修订：
  ① generations.json 是随机采样参考而非贪心参考，一致性条款已改用规划层
  独立生成的贪心 fixture；② 白名单补入规划层提供的两个 fixture 文件。
  另确认 kaleido `src/kaleido/lm/` 在工件生成 revision（cee0ef8）与当前
  HEAD（b3b9b4a）之间无改动，架构语义兼容；MANIFEST 需同时记录两个 revision。

## 1. 研究问题

P0 之后所有"优化收益"的测量都需要一个固定、可信的被测件（SUT）。
本任务回答：prism 如何以可校验完整性的方式接入 kaleido 已验证的导出模型，
并保证 prism 侧的前向/解码结果与 kaleido 参考逐 token 一致。

可能的失效形式：

- 工件在复制/加载中被污染或版本漂移（对策：sha256 清单校验，加载即校验）。
- prism 重实现的前向与 kaleido 有隐性差异（如 LayerNorm eps、掩码、GELU 近似），
  导致后续基准测的是"另一个模型"（对策：贪心解码 token 序列与参考逐位一致）。
- 参考路径硬编码本机绝对路径，仓库不可移植（对策：工件复制入 prism 仓库，
  来源与哈希记入清单）。

## 2. 改动白名单

允许修改/新增以下路径，其余一律禁止：

- 新增 `src/prism/sut.py`
- 新增 `tests/test_sut.py`
- 新增 `data/sut/kaleido-p3/` 目录，内容仅限：
  - `model-42.npz`（从 kaleido 复制的权重）
  - `tokenizer.json`（从 kaleido 复制的分词器）
  - `MANIFEST.json`（见实现要求 3）
- 修改 `src/prism/cli.py`（仅当需要挂接 `prism sut check` 子命令时）
- 修改 `tests/test_cli.py`（仅当 CLI 输出变化导致的最小适配）

**规划层已提供的 fixture**（已入库，只读使用，禁止重新生成或修改）：

- `data/sut/kaleido-p3/reference_greedy.json`——贪心解码 golden 参考，
  由规划层用 kaleido 原生 PyTorch 模型独立生成并自验（含 provenance）
- `data/sut/kaleido-p3/reference_logits.npz`——within_window 完整 50 token
  序列的逐步 logits 对照（float32，CPU）

**只读参考**（允许读、禁止改、禁止在 prism 代码中 import）：

- `/Users/wangchen/Developer/kaleido/results/p3/reference/`（工件来源）
- `/Users/wangchen/Developer/kaleido/src/kaleido/lm/`（架构与分词器实现参考）

## 3. 禁止事项

- 修改 `pyproject.toml`、`uv.lock`；引入任何新第三方依赖（仅标准库 + numpy）
- 修改 kaleido 仓库的任何文件；`git push`
- 删除 prism 现有文件；改动测量协议（T3 范畴，本任务不做任何计时/统计）
- 把 kaleido 作为 Python 依赖 import——prism 必须用自有 NumPy 代码实现前向

## 4. 实现要求

1. **工件落地**：从 `/Users/wangchen/Developer/kaleido/results/p3/reference/`
   复制 `model-42.npz`、`tokenizer.json` 到 `data/sut/kaleido-p3/`；
   生成 `MANIFEST.json`，逐文件记录：来源绝对路径、来源 sha256、
   kaleido 仓库当前 commit hash（`git -C <kaleido> rev-parse HEAD`）、复制日期。
2. **加载与校验**：`src/prism/sut.py` 提供
   `load_sut(path: Path) -> SUT`，加载时：先按 `model-42.json`（与 npz 同源的
   kaleido 数组哈希清单，复制时一并参考其内容，可将数组哈希并入 MANIFEST.json）
   校验每个数组的 sha256，不符即抛错（fail-closed）。
   模型配置（layers=2, heads=4, width=128, vocab=512, ctx=128）从工件读取，
   不硬编码。
3. **前向与解码**：纯 NumPy 实现该架构的前向（因果掩码注意力、LayerNorm、
   MLP，结构以 kaleido `lm/model.py` 为准）与贪心解码；KV cache 暂不需要
   （P1 范畴），允许 O(n²) 朴素实现。
4. **一致性自证**（v2 修订，替代 v1 的 generations.json 条款——
   原条款误将 temperature=0.8 的随机采样参考当作贪心参考，已由规划层修正）：
   - 对 `reference_greedy.json` 的两个用例（within_window：18 token prompt
     贪心生成 32 token；window_rollover：127 token prompt 贪心生成 16 token；
     两者均禁用 EOS 停止，滑窗重编号规则见 fixture provenance）：
     **prism 生成的完整 token id 序列必须与 fixture 完全一致**（逐 token，长度一致）。
   - 对 `reference_logits.npz`：prism 对同一 50 token 序列一次前向，
     报告与 fixture logits 的最大绝对误差（作记录，不作门槛）；
     每个生成位置的 argmax 必须与次一 token 一致。
   - kaleido `generations.json` 的三条采样记录仅用于校验分词器与结构
     （`[256] + encode(prompt) == prompt_ids`），不得用于逐 token 生成比对。
   - 一致性检查经 `prism sut check` 暴露，不一致时以非零状态退出。
5. 代码风格：ruff line-length 100、全类型注解、docstring 与现有文件一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check    # 若实现了该子命令
git diff --stat                    # 改动不得越出白名单
```

人工检查项：MANIFEST.json 中的 sha256 与 kaleido 来源文件实测一致；
故意篡改一个数组后 `load_sut` 必须报错（规划层抽查）。

## 6. 汇报要求

1. 实际改动/新增文件清单（逐文件）
2. 每条验收命令的实际输出摘要
3. token 级一致性证据（样本数、通过数、logits 最大绝对误差）
4. 与规格的任何偏差及原因（无偏差写"无"）

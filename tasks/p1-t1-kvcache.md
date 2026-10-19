# 任务规格 P1-T1：KV cache 推理回路

- 阶段：P1 推理计算（docs/ROADMAP.md）
- 分支：`codex/p1-kvcache`
- 风险等级：**高**（核心推理回路，后续所有 cache 实验的地基）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。

## 1. 研究问题

P0 的 SUT 是无 cache 的 O(n²) 朴素解码。本任务回答：
KV cache 在**保持逐 token 输出完全一致**的前提下，能给 decode 带来多大
加速、付出多少内存代价？cache 语义（窗口滚动时的重建）如何保证正确？

可能的失效形式：

- cache 实现与 naive 有隐性差异（位置编号、掩码、窗口滚动边界），
  导致后续所有 cache 实验测的是错误回路（对策：与 P0 fixture 逐 token 比对）。
- 只在小窗口内验证，跨窗口行为未覆盖（对策：window_rollover 用例强制通过）。
- 加速比无测量协议锚定（对策：用 P0 measure()、附 env 指纹与 T4 阈值）。

## 2. 改动白名单

- 新增 `src/prism/kvcache.py`
- 修改 `src/prism/sut.py`（仅挂接 cache 路径，不得改变 naive 路径行为）
- 修改 `tests/test_sut.py` 或新增 `tests/test_kvcache.py`
- 新增 `configs/p1_kvcache.json`（cache 对照测量配方）

## 3. 禁止事项

- 修改 `measure.py`、`env.py`、`record.py`、`data/sut/` 下任何文件
- 改依赖、`git push`、删除文件；实现分页/驱逐（P1-T2 范畴）
- 改变 naive 前向的任何数值行为（P0 一致性测试必须原样通过）

## 4. 实现要求

1. **cache 语义**（以 kaleido `lm/model.py`、`lm/inference.py` 为只读参考，
   但用 prism 自有 NumPy 实现）：
   - 每层维护 K/V cache，decode 每步只对新 token 做前向，
     attend 范围为 cache 全部内容；
   - 序列长度超过 `context_length` 时按滑窗规则处理：窗口滚动后
     对当前窗口**重建 cache**（记录重建次数），位置从 0 重编号，
     与 naive 的窗口语义严格一致；
   - `generate(..., use_cache=True)` 与 `use_cache=False` 并行存在，
     naive 路径保持 P0 行为不变。
2. **正确性门槛（硬）**：对 `data/sut/kaleido-p3/reference_greedy.json`
   两个用例，cached 生成的完整 token id 序列必须与 naive/fixture 逐位一致
   （within_window 50/50、window_rollover 143/143，后者验证窗口滚动重建）。
   另加随机 prompt 的 cached-vs-naive 一致性测试（固定种子，多组长度，
   含 127/128/129 边界）。
3. **性能对照（记录，不设门槛）**：`configs/p1_kvcache.json` 配方，
   用 P0 `measure()` 对 decode 32 token 负载分别测 naive 与 cached
   （warmup/repeat 参照 T4 结论设定），经 `save_run()` 落盘 runs/，
   汇报加速比与 peak_tracemalloc_bytes 差；低于 0.092584 ms 的差异
   标注不可分辨。
4. cache 内存占用给出解析推算（层数 × 头数 × 窗口 × 头维 × 2 × 4B）
   并与实测对照，差异需解释。
5. 风格：ruff line-length 100、全类型注解、docstring 与现有文件一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check    # P0 一致性必须原样通过
git status --short
```

人工检查项：cached/naive 输出逐 token 一致（规划层另起进程抽查）；
cache 重建次数在 window_rollover 用例中非零。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. cached vs naive 的测量对照（p50/p95/p99、加速比、内存差、env 指纹、
   config 哈希、runs 目录）
4. 与规格的任何偏差及原因（无偏差写"无"）

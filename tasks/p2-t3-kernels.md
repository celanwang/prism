# 任务规格 P2-T3：算子级优化逐项消融

- 阶段：P2 推理优化（docs/ROADMAP.md）
- 分支：`codex/p2-kernels`
- 风险等级：低（每项消融独立、对照明确；正确性 harness 已存在）
- 依赖：P1-T1 已合并

## 1. 研究问题

"算子级优化的单项贡献各是多少"：
对 prism 的 NumPy 前向，以下三项各自带来多少时延变化——
① 合并 QKV 一次矩阵乘 vs 三次独立乘（现有实现即为合并式，
   反向消融：拆开后慢多少）；
② attention 分数计算的内存布局（heads-first vs batch×time 主序
   reshape 路径）；
③ MLP 两次矩阵乘的中间缓冲复用。
每项独立测量、独立结论，回答"哪些微优化值得做、哪些是噪声"。

可能的失效形式：

- 多项优化打包提交，无法归因（对策：每项消融独立配方、
  独立 runs，禁止合并测量）。
- 微优化幅度低于测量器分辨率被当作收益（对策：T4 阈值判定
  为强制结论项，"不可分辨"是合法答案）。
- 消融改变数值（对策：每项过同一 fixture 逐 token 门槛）。

## 2. 改动白名单

- 新增 `src/prism/ablating.py`（可切换的前向变体，不动现有路径）
- 新增 `tests/test_ablating.py`
- 新增 `configs/p2_kernels.json`
- 新增 `scripts/kernel_bench.py`
- 新增 `runs/<日期>-p2-kernels-*/` 产物目录
- 新增 `results/p2/kernels.md`（报告草稿）

## 3. 禁止事项

- 修改 `sut.py`、`kvcache.py` 等现有模块（变体放新文件）
- 改依赖、`git push`、删除文件；打包测量多项消融
- 把低于 T4 阈值的差异表述为收益

## 4. 实现要求

1. **变体实现**：每项消融一个可开关的前向变体
   （与基线共享大部分代码，开关处明确注释），
   全部过 fixture 两用例逐 token 一致门槛。
2. **逐项测量**：`configs/p2_kernels.json` 为每项定义独立配方
   （prefill T=128 与 decode 32 token 两负载，warmup/repeat
   参照 T4），每项独立 save_run()。
3. **报告草稿**：每项消融的 p50/p95/p99 对照、T4 阈值判定、
   结论（值得/不值得/不可分辨）及原因分析。
4. 测试：变体正确性（逐 token）、配方加载、阈值判定逻辑；
   不断言性能方向。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check
uv run --locked python scripts/kernel_bench.py --check
git status --short
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 三项消融的对照数据与逐项结论（含 env 指纹、config 哈希）
4. 与规格的任何偏差及原因（无偏差写"无"）

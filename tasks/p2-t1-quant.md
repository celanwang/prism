# 任务规格 P2-T1：INT8/INT4 权重量化与精度–时延–内存三角

- 阶段：P2 推理优化（docs/ROADMAP.md）
- 分支：`codex/p2-quant`
- 风险等级：**高**（改动数值路径，精度退化的判定协议本身需谨慎）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 依赖：P1-T1 已合并（cached 回路作为量化后 decode 的承载）

## 1. 研究问题

"权重量化在 CPU 上的精度–时延–内存三角长什么样"：
对 prism 的 SUT，per-channel INT8 与 group-wise INT4 权重量化
（仅权重、激活保持 float32）分别把权重内存降到多少、
decode 时延变化多少、输出精度退化多少？**在 0.5M 小模型上
量化噪声是否大到吞掉内存收益**——这是小模型特有的真问题。

可能的失效形式：

- 精度退化用 logits 误差单一指标误判（对策：三指标并用——
  逐 token 一致率、logits 最大/均方误差、fixture 贪心序列分歧位置）。
- 时延收益被反量化开销吞掉却未分解（对策：分别记录矩阵乘与
  反量化耗时占比，measure() 协议测量）。
- 量化实现改变语义（对策：非量化路径零改动，P0/P1 测试原样通过）。

## 2. 改动白名单

- 新增 `src/prism/quant.py`
- 新增 `tests/test_quant.py`
- 新增 `configs/p2_quant.json`（量化方案与测量配方）
- 新增 `scripts/quant_bench.py`
- 新增 `runs/<日期>-p2-quant-*/` 产物目录
- 新增 `results/p2/` 目录（本任务报告草稿，规划层终审）

## 3. 禁止事项

- 修改 `sut.py`、`kvcache.py`、`measure.py`、`env.py`、`record.py`
- 改依赖、`git push`、删除文件；量化激活或 KV cache（后续任务范畴）
- 改变非量化路径任何数值行为

## 4. 实现要求

1. **量化方案**（仅 Linear 权重，含 embedding 与否需配置并说明）：
   - INT8 per-output-channel 对称量化（scale 每行一个）
   - INT4 group-wise 对称量化（group size 可配，如 32/64/128）
   - 推理时反量化为 float32 再矩阵乘（CPU 上无原生 INT MAC，
     如实标注：本阶段测的是内存收益与反量化开销，非 INT 加速）
2. **精度对照**（对 fp32 基线）：
   - fixture 两贪心用例的逐 token 一致率与首个分歧位置
   - reference_logits.npz 的 logits 最大/均方误差
   - 随机 prompt（固定种子）的 token 一致率统计
3. **内存对照**：量化后权重字节数精确计算（含 scale/zero-point
   簿记），与 fp32 基线的压缩比；measure() 峰值旁证。
4. **时延对照**：`configs/p2_quant.json` 配方测 prefill/decode
   各方案 p50/p95/p99，经 save_run() 落盘；附 T4 阈值判定。
5. 报告草稿 `results/p2/quant.md`：三角数据表 + 明确边界声明
   （小模型、CPU、仅权重量化、反量化路径）。
6. 风格：ruff line-length 100、全类型注解、docstring 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check
uv run --locked python scripts/quant_bench.py --check
git status --short
```

人工检查项：INT8 round-trip（量化→反量化）误差有界验证；
fp32 路径输出与量化模块引入前完全一致。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 三角数据全表（精度三指标 × 内存压缩比 × 时延，含 env 指纹、
   config 哈希、runs 目录）
4. 与规格的任何偏差及原因（无偏差写"无"）

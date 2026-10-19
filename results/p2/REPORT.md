# P2 推理优化：阶段验收报告

研究问题（ROADMAP P2）："权重量化在 CPU 上的精度–时延–内存三角长什么样？
投机采样的接受率–收益曲线何时值得？"
本报告为阶段汇总；各任务的完整数据、runs 指针与环境锚点见三个分项报告。

## 交付物与证据指针

| 任务 | 交付物 | 验收证据 |
|---|---|---|
| T1 权重量化 | `src/prism/quant.py`、[quant.md](quant.md) | [PR #11](https://github.com/celanwang/prism/pull/11)；477 tests |
| T2 投机采样 | `src/prism/speculative.py`、[speculative.md](speculative.md) | [PR #14](https://github.com/celanwang/prism/pull/14)；604 tests；80 组逐 token 硬门槛 |
| T3 算子消融 | `src/prism/ablating.py`、[kernels.md](kernels.md) | [PR #16](https://github.com/celanwang/prism/pull/16)；533 tests；24 组独立 runs |

全部测量遵循 P0 协议（measure()、save_run()、env 指纹
`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`、
T4 分辨率阈值 0.092584 ms）。

## 主要结论

1. **权重量化（CPU、仅权重、反量化路径）**：INT8 per-channel 全模型压缩
   2.57× 且 fixture 逐 token 一致 50/50；INT4 group-wise 压缩约 3.40×
   但小模型上退化明显（fixture 一致率降至 29/50）。反量化开销使
   "INT 加速"在本环境不成立——量化收益仅限内存侧。
2. **投机采样（INT8 草稿 + fp32 验证）**：工作负载接受率=1（同源小模型），
   但所有 k（1/2/4/8）端到端均慢于 fp32/INT8 基线；解析模型证实
   D/T≈1.4–1.5、C/T≥k+1，[0,1] 区间无可行盈亏平衡点。
   端到端实测与解析预测互证的负结果。
3. **算子消融（双跑一致才保留结论）**：QKV 合并在 decode 上值得保留
   （拆分慢约 1.5–1.8 ms，超 T4）；attention 布局整理在 decode 上不值得；
   MLP 缓冲复用不可分辨。prefill 侧全部不可分辨。

## 边界声明

- 全部数字为 Apple M4 Max / CPU / 单线程实测；小模型（0.48M 参数）
  结论不可外推到大模型（同源草稿接受率虚高已明确标注）。
- 精度"实现通过"不等于"精度可接受"；P2 未设精度合格阈值。
- T4 阈值为经验扰动尺度，超过它不自动证明显著性；未估计置信区间。

## 负结果与过程记录

分页无收益（P1）、投机采样无盈亏点、INT4 退化、MLP 复用不可分辨
均为合法产出并完整保留原始 runs。执行过程中 Codex 额度中断一次，
中断现场经 resume 无损续接；额度恢复后全程串行执行。

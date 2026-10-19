# 任务规格 P4-T1：并行策略显存–通信解析模型

- 阶段：P4 训练系统（模拟为主，docs/ROADMAP.md）
- 分支：`codex/p4-models`
- 风险等级：低（纯解析/模拟，无实测硬件依赖）
- 依赖：P0 记录规范；与 P1–P3 无代码耦合，可随时插入

## 1. 研究问题

"给定显存预算，并行策略与 micro-batch 的解析预测与实测差多少"
的解析半边：对 transformer 训练，数据并行（DP）、张量并行（TP）、
流水线并行（PP）及 ZeRO-1/2/3 的显存占用与每步通信量的
解析模型是什么？模型的哪些假设（激活重算、梯度累积、
通信–计算重叠）主导误差？

可能的失效形式：

- 公式照抄论文但变量定义含糊（对策：每个符号有定义、
  每个公式有量纲检查测试）。
- 解析模型被当作实测结论（对策：所有输出标注"解析模型"，
  实测校准在 P4-T2）。

## 2. 改动白名单

- 新增 `src/prism/parallel_models.py`
- 新增 `tests/test_parallel_models.py`
- 新增 `configs/p4_models.json`（模型规格与策略网格）
- 新增 `scripts/parallel_scan.py`
- 新增 `runs/<日期>-p4-models-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 现有模块；改依赖、`git push`、删除文件
- 把解析预测表述为实测；引入真实多卡假设的硬件数字而不标注

## 4. 实现要求

1. **解析模型**（输入：模型规格（layers/hidden/ffn/vocab/seq）、
   并行策略（DP/TP/PP 度数）、micro-batch、精度字节数；
   输出：每设备显存分解（参数/梯度/优化器状态/激活）、
   每步每设备通信量（all-reduce/all-gather/p2p 分项）、
   静态吞吐推算（注明不含通信–计算重叠与 kernel 效率）：
   - DP + ZeRO-1/2/3 的显存分摊公式
   - TP 的激活复制与 all-reduce 量（按 Megatron 切分规则）
   - PP 的 bubble 比例（(p-1)/(m+p-1)，m 为 micro-batch 数）
2. **量纲与守恒测试**：字节量纲一致；ZeRO-3 全集群总显存
   ≈ 单卡未分摊总量（无免费午餐）；PP bubble 在 m→∞ 时→0。
3. **扫描**：`configs/p4_models.json` 定义模型规格网格
   （以 prism SUT 的 0.5M 与一个假想的 7B 规格两档，
   7B 明确标注假想）、策略网格，输出显存–通信表。
4. 结果经 save_run() 落盘（标注"解析模型"）。
5. 风格：ruff line-length 100、全类型注解、docstring 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/parallel_scan.py --check
git status --short
```

人工检查项：规划层抽查两个公式的推导（ZeRO-2 显存、PP bubble）
与文献一致。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 代表性格网点的显存–通信表（标注解析模型）
4. 与规格的任何偏差及原因（无偏差写"无"）

# 任务规格 P0-T4：测量工具自身开销标定

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-calibrate`
- 风险等级：低（规格 → 执行 → 验收）
- 依赖：T3 已合并（`src/prism/measure.py`）

## 1. 研究问题

T3 的测量器自身会扰动被测对象：tracemalloc tracing 拖慢计时、
样本簿记引入分配。本任务回答：measure() 的自身开销有多大，
多小的"优化收益"是测量器无法分辨的（分辨率下界）？
没有这个数字，P1/P2 的小幅收益无法判断真伪。

可能的失效形式：

- 开销随被测函数分配行为变化，单一数字误导（对策：空函数/纯计算/
  分配密集三类探针分别标定）。
- 标定结论无环境锚定（对策：报告强制携带 env 指纹与 config 哈希，
  复用 T1/T3 机制）。

## 2. 改动白名单

- 新增 `scripts/calibrate.py`
- 新增 `tests/test_calibrate.py`
- 新增 `results/p0/calibration.md`（由脚本生成的标定报告，人工核对后入库）
- 新增 `configs/calibrate.json`（标定配方：warmup/repeat/探针定义）

## 3. 禁止事项

- 修改 `src/prism/measure.py`、`src/prism/env.py`（测量器本身不许在本任务中调整，
  发现测量器缺陷时停下汇报）
- 改依赖、`git push`、删除文件
- 把标定数字写成"性能结论"——本任务只描述测量器自身，不测任何 SUT

## 4. 实现要求

1. `scripts/calibrate.py` 用 T3 的 `measure()` 标定三类探针
   （探针参数入 `configs/calibrate.json`，不在代码里写死）：
   - 空操作探针：`lambda: None`，给出计时+tracing 的固定开销基线
   - 纯计算探针：无分配的算术闭包，给出典型 CPU 负载下的扰动
   - 分配密集探针：每次调用分配固定数量小对象，给出 tracemalloc
     追踪成本随分配量的放大系数
2. 每个探针输出：带 tracing 的测量结果 vs 不带 tracing 的裸计时对照
   （裸计时用 perf_counter_ns 直接测，不经 measure()），
   开销 = 两者均值差与分位数差。
3. 报告 `results/p0/calibration.md` 包含：env 指纹、config 哈希、
   每个探针的 p50/p95/p99 对照表、结论段（测量器可分辨的最小收益量级，
   以及"低于 X ms 的差异不作为证据"的建议阈值）。
4. 脚本可重复运行（幂等覆盖报告），`--check` 模式只打印不写入。
5. 测试：探针行为正确（如分配探针确实分配）、报告包含必需字段、
   `measure()` 被以规格化参数调用（mock 验证），不断言具体性能数字。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/calibrate.py --check
git status --short   # 改动不得越出白名单
```

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 三类探针的开销数字（含 env 指纹）与建议分辨率阈值
4. 与规格的任何偏差及原因（无偏差写"无"）

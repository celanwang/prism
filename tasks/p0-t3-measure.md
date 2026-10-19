# 任务规格 P0-T3：测量协议核心（measure API）

- 阶段：P0 测量地基（docs/ROADMAP.md）
- 分支：`codex/p0-measure`
- 风险等级：**高**（全项目的度量衡，协议一旦定错所有下游数字作废）
- 门控流程：双阶段——先 `codex exec -s read-only` 输出实施计划，
  规划层批准后才放行 workspace-write 执行。

## 1. 研究问题

"本机推理的时延/吞吐/内存如何可信测量"的核心子问题：
什么样的测量协议产出的数字才有资格成为"证据"？
本任务实现 prism 唯一的测量入口，后续所有阶段（P1 内存–吞吐曲线、
P2 量化收益、P3 尾时延）只允许通过它产出性能数字。

可能的失效形式：

- 无预热导致首样本偏置（JIT/缓存/惰性初始化污染均值）。
- 只报均值不报分布，尾时延被掩盖（对策：样本全量保留 + 分位数）。
- 计时分辨率与统计方法未指定，不同任务实现漂移（对策：本模块为唯一实现，
  分位数插值方式写死并测试）。
- 结果与配置、环境脱钩，无法复现（对策：每条记录强制携带 config 哈希与
  env 指纹，复用 T1 的 `prism.env`）。

## 2. 改动白名单

- 新增 `src/prism/measure.py`
- 新增 `tests/test_measure.py`
- 修改 `src/prism/cli.py`（仅当需要挂接演示子命令时，可选）
- 修改 `tests/test_cli.py`（仅 CLI 输出变化的最小适配）

## 3. 禁止事项

- 修改 `pyproject.toml`、`uv.lock`；新第三方依赖（仅标准库 + numpy）
- 修改 `src/prism/env.py`、`src/prism/sut.py`（只可 import，不可改）
- `git push`；删除文件；实现任何被测业务逻辑（本任务只有测量器）

## 4. 实现要求

1. **API**：`measure(fn: Callable[[], T], *, warmup: int, repeat: int,
   config: dict) -> Measurement`
   - 先调用 `fn()` `warmup` 次（结果丢弃），再计时调用 `repeat` 次；
   - 每次计时用 `time.perf_counter_ns()`，样本以毫秒浮点记录；
   - `warmup >= 0`、`repeat >= 1`，参数非法抛 `ValueError`。
2. **统计**（`Measurement` 字段，dataclass，全类型注解）：
   - `samples_ms: list[float]`（全量保留）
   - `mean / stdev`（样本标准差，ddof=1；repeat=1 时 stdev 为 0.0）
   - `p50 / p95 / p99`：numpy `np.percentile(..., method="linear")`，
     插值方式写死在实现里并在 docstring 注明
   - `peak_tracemalloc_bytes`：用 `tracemalloc` 包住计时区段取峰值；
     tracemalloc 的开销影响记入 docstring（T4 标定的对象）
   - `max_rss_bytes`：计时区段结束时 `resource.getrusage(RUSAGE_SELF).ru_maxrss`
     （注明：macOS 上单位是字节，且为进程高水位，语义仅作参考）
   - `config_hash`：`config` 按 key 排序 JSON 的 sha256
   - `env_fingerprint`：调用 T1 的 `prism.env.collect_env()` +
     `env_fingerprint()`，复用，不重新实现
   - `captured_at`：ISO 8601
3. **序列化**：`Measurement.to_json() -> str`（sort_keys、缩进 2）与
   `Measurement.from_json(str)`，往返一致。
4. **测试**至少覆盖：
   - 预热次数生效（用计数闭包断言 fn 总调用数 = warmup + repeat）
   - 参数非法抛错；repeat=1 时 stdev 为 0.0
   - 分位数约定：对已知样本序列断言 p50/p95 精确值（锁定插值方式）
   - to_json/from_json 往返一致；env_fingerprint 与 config_hash 字段存在
   - 计时单调性不测绝对值（禁止 wall-clock 断言导致 flaky），
     只测样本数量与非负性
5. 风格：ruff line-length 100、全类型注解、docstring 与现有文件一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
git diff --stat   # 不得越出白名单
```

人工检查项：同一代码路径两次运行产出的 Measurement，
`config_hash` 与 `env_fingerprint` 字段必须相同（规划层抽查）。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令的输出摘要
3. 分位数插值方式、stdev 自由度的选择说明
4. 与规格的任何偏差及原因（无偏差写"无"）

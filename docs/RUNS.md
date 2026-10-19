# 实验记录

每次运行保存为 `runs/<YYYYMMDD>-<slug>/`，日期采用创建时的 UTC 日期；
slug 使用字母、数字，以 `-` 或 `_` 分隔。目录内必须四件齐全：

| 文件 | 字段语义 |
| --- | --- |
| `config.json` | 任意 JSON 对象，保留设备、预热/重复次数等实验配置；拒绝 NaN/Inf。配置按 T3 的 `json.dumps(sort_keys=True, allow_nan=False)` 编码后计算 SHA-256。 |
| `measurement.json` | T3 `Measurement.to_json()` 输出；时间单位 ms，内存单位 bytes，包含原始样本、分位数、配置哈希及环境指纹。 |
| `env.lock` | 保存时 `collect_env()` 的完整快照；复用 `env_fingerprint()`，仅排除采集时间，其余字段参与指纹计算，必须与 measurement 一致。 |
| `meta.json` | `created_at` 为带时区的 UTC 创建时间，`prism_version` 为保存工具版本；`task`/`phase` 从 config 同名字段复制，未提供时为 null。 |

保存前验证配置哈希及环境指纹。先写同一父目录中的隐藏临时目录，再一次 rename
发布，失败清理临时目录；目录锁串行化保存者，已存在的目录（包括空目录）一律拒绝
覆盖。同一 UTC 日期同名 slug 不能重复；不同日期可复用。原子发布不承诺断电持久性。
加载必须检查全部文件、JSON 结构、必需字段及哈希/指纹；缺项、不一致立即报错。
哈希交叉校验用于发现记录损坏，不提供防止攻击者同时重写所有文件的数字签名。

```bash
# measurement 必须由 T3 测量工具提供，保存命令不执行测量
prism record save --slug baseline --config config.json < measurement.json
prism record save --slug baseline --config config.json --measurement measurement.json
prism record summary
prism record summary --runs-dir /path/to/runs
```

`--runs-dir` 默认是当前工作目录下的 `runs/`，save/summary 均可指定。
summary 按目录名排序，输出 slug、p50/p95/p99、两种峰值内存、完整环境指纹和
配置哈希前 8 位；空目录或尚不存在的目录输出空表。隐藏临时目录不作为记录读取；
任何可见的损坏记录均使汇总失败，不静默跳过。

**`results/` 只允许由工具从 `runs/` 生成，禁止手改。** 报告生成应复用 `summarize()`，
先验证所有输入再发布输出。正式性能结论必须注明硬件/设备、预热和重复次数、分位数、
配置哈希；smoke test 不算实验证据，模拟结果不得写成实测结论。

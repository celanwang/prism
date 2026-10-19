# P4 训练系统：解析模型与单机小规模实测校准

本阶段在可测范围内回答：P4-T1 的常驻状态预测在同精度下能否对齐实际数组，
激活是否随 micro-batch 线性增长，以及绝对预测误差来自哪些假设。
SUT 的三点校准中，参数/梯度/优化器存储逐项对齐；层内保留激活呈 1:2:4，
但原解析模型高估 27.4005%。这验证了本实现的批大小比例关系，不能据此验证多卡策略。

## 解析模型要点

P4-T1 实现见 [parallel_models.py](../../src/prism/parallel_models.py)，
扫描入口见 [parallel_scan.py](../../scripts/parallel_scan.py)，
配方见 [p4_models.json](../../configs/p4_models.json)。该模型面向 tied embeddings、
learned positions、含 bias 和 LayerNorm 的 dense GPT，所有计算均为解析值。

- 参数量：`P=Vh+Ch+2h+L(4h²+2hf+9h+f)`；SUT 为 **478720** 个标量。
- 令 `q=P/(TP×PP)`，每设备参数/梯度/Adam 存储分别为 `q×w`、`q×g`、`q×o`，
  ZeRO-1 分片优化器，ZeRO-2 再分片梯度，ZeRO-3 再分片参数；分片因子为 DP。
  这些是理想均衡放置下的常驻量，未计 live gathers、workspaces、碎片或临时梯度。
- 不重算时每层、每 micro-batch 激活为
  `A=S×b×[h(4a+2k)+(4ah+2af+(2a+k)HS)/TP]`。
  `a` 是浮点激活字节，`k` 是 dropout mask 字节；全层重算时改用 `S×b×h×a`。
  flush 1F1B 最坏 stage 保留量近似为 `(L/PP)×min(m,PP)×A`，不是设备峰值。
- 环形 all-reduce 每 rank 发送 `2(n−1)/n×payload`，all-gather/reduce-scatter
  为其一半。DP/ZeRO 依据梯度归约与参数收集次数累加；TP 每层每 micro-batch
  4 次 all-reduce（全重算 6 次）；PP 每边界发送一次前向与一次反向激活。
  只计发送；PP 同时给出 stage 均值和逐 stage 量，没有网络耗时或 overlap 模型。
- PP 气泡占 elapsed 的比例为 `(PP−1)/(m+PP−1)`；计算工作量
  `F=6PN+12LNS h`，`N=DP×b×m×S`，全层重算乘 `4/3`。
  `F/[DP×TP×PP×c×(1−bubble)]` 使用显式假想算力 `c`，不代表硬件训练性能。

本次原样调用 P4-T1 的 `ModelSpec/Strategy/Precision/predict`，未修改其公式。
DP=TP=PP=1、ZeRO=0、m=1、不重算。FP32 权重/梯度/激活各 4 B，
Adam moments 共 8 B/参数，mask 配置沿用 1 B。
P4-T1 默认的 FP16 与含 master copy 的 12 B optimizer 配置不直接用于 FP32 对照。

## 校准数据

续跑时修复了 JSON 重载后 Markdown 字段顺序不稳定的问题，并重跑正式校准；
以下数字来自最终源码。早期 `102056043664` 正式 run 和 `102330548159-check` smoke run
仅作历史记录，早期源码保存在 `runs/20261019-p4-calibrate-102056043664-b1/source_snapshot.py`，
其哈希与早期 config 中记录的脚本哈希一致。

选择规格允许的 **prism 自有 NumPy 真实训练步**，没有使用训练记录回放。
从完整性校验后的 `data/sut/kaleido-p3` 加载实际权重，每个 B 独立复制起始状态。
模型为 L=2、h=128、f=512、V=512、H=4、位置容量 C=128，实测 S=128。
前向为 causal attention、pre-LN、tanh GeLU、tied output projection，dropout=0；
反向显式传播 mean cross entropy 到每个参数；AdamW 做 bias correction 和解耦权重衰减。
只读参考 kaleido 的 Tensor backward 与优化器实现，运行时不依赖或导入 kaleido。

硬件/设备：**Apple M4 Max，CPU / arm64，16 逻辑 CPU，48 GiB RAM**；
macOS 15.8、Python 3.12.15、NumPy 2.5.4。CLI 在加载 NumPy 前设置五个 BLAS/OpenMP
线程环境变量为 1。完整环境见各 run 的 `env.lock`。

每点先执行 1 次未计时存储诊断训练步，再预热 **5** 步，测量 **30** 步；
步内含前向、CE、反向、AdamW 更新及临时对象释放。模型和梯度/moments 预分配，
数据生成、加载、存储清点及文件写入不计时。输入是 seed=42 的固定随机 next-token
序列，权重逐步更新而非每个样本复位；这是一项训练系统校准，不是学习质量实验。
p50/p95/p99 使用 NumPy linear 插值，保留全部原始计时样本。

配方 SHA-256：`7ce69f0b052305620caedce32d8bbe687f47c17b0d6b9541ecc1a9daad2af0ca`。
环境指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。
采集时间：2026-10-19 11:44:18 UTC。

各点通过 `measure()` 测量、`save_run()` 落盘并以 `load_run()` 校验。
正式产物目录如下；每点有 config、measurement、env.lock、meta、calibration_point，
B=1 额外包含全表 [calibration.json](../../runs/20261019-p4-calibrate-114418775926-b1/calibration.json)
和 [calibration.md](../../runs/20261019-p4-calibrate-114418775926-b1/calibration.md)。

| B | run 目录 | 测量配置 SHA-256 |
| ---: | --- | --- |
| 1 | `20261019-p4-calibrate-114418775926-b1` | `1c52b92b6262333b47d4ead385059223ba5f9e15e09ab946f69851230120f814` |
| 2 | `20261019-p4-calibrate-114418775926-b2` | `0cd65103748f9b13cb5e2146e1ffa7bf802f74e2b216a86859449d1d37c9d35d` |
| 4 | `20261019-p4-calibrate-114418775926-b4` | `7023c6798f6825b1be749a4cd8ca6beba5fa1e9b109cfd0c3e60d3ff42067290` |

以下全部为 bytes。Δ=解析−实测，绝对偏差=|Δ|，相对偏差=Δ/实测；
`total` 只加常驻状态与层内保留激活，不是进程/设备峰值。
实测存储来自真实分配的 ndarray owning buffers 的 `nbytes`，不是由 P4 公式回填。
Q/K/V 的 views 共用 QKV 底层存储，按 owning allocation 去重。

| B | 项目 | 解析 B | 实测 B | Δ B | 绝对偏差 B | 相对偏差 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 参数 | 1914880 | 1914880 | 0 | 0 | 0% |
| 1 | 梯度 | 1914880 | 1914880 | 0 | 0 | 0% |
| 1 | 优化器 | 3829760 | 3829760 | 0 | 0 | 0% |
| 1 | 激活 | 3342336 | 2623488 | +718848 | 718848 | +27.4005% |
| 1 | total | 11001856 | 10283008 | +718848 | 718848 | +6.9906% |
| 2 | 参数 | 1914880 | 1914880 | 0 | 0 | 0% |
| 2 | 梯度 | 1914880 | 1914880 | 0 | 0 | 0% |
| 2 | 优化器 | 3829760 | 3829760 | 0 | 0 | 0% |
| 2 | 激活 | 6684672 | 5246976 | +1437696 | 1437696 | +27.4005% |
| 2 | total | 14344192 | 12906496 | +1437696 | 1437696 | +11.1393% |
| 4 | 参数 | 1914880 | 1914880 | 0 | 0 | 0% |
| 4 | 梯度 | 1914880 | 1914880 | 0 | 0 | 0% |
| 4 | 优化器 | 3829760 | 3829760 | 0 | 0 | 0% |
| 4 | 激活 | 13369344 | 10493952 | +2875392 | 2875392 | +27.4005% |
| 4 | total | 21028864 | 18153472 | +2875392 | 2875392 | +15.8394% |

`measure()` 的 tracemalloc 峰值是测量区间的增量分配峰值，排除已预分配常驻状态；
不覆盖所有 native 分配。RSS 为 macOS 的进程生命周期高水位，三个 B 同进程顺序测量，
不能解释成各自训练步新增内存，也不能按 B 相减。全前向保留额外包含 final norm、
head input、logits，单列以免与 P4 层内口径混淆。

| B | p50 ms | p95 ms | p99 ms | traced peak B | lifetime RSS B | 全前向保留含 logits B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.051125 | 5.239298 | 5.353562 | 5847848 | 67649536 | 3017216 |
| 2 | 7.713604 | 7.983909 | 8.218588 | 11683187 | 109019136 | 6034432 |
| 4 | 13.280354 | 14.050111 | 14.516120 | 23351243 | 205488128 | 12068864 |

每点共执行 36 次优化器更新；B=1/2/4 诊断 loss→末步 loss 分别为
12.102966→5.483827、12.478195→5.810217、12.259622→6.050737，状态均有限。
loss 只作为真实训练执行的辅助证据。正式数据与 `--check` 的 smoke 产物分开标记。

## 激活线性性检验

从实际缓存清点得到 `2623488 / 5246976 / 10493952 B`，除以 B 后均为 **2623488 B**。
以 B=1 为锚点，B=2/4 的比例预测残差均为 0 B，相对误差 0，满足 1e−6 阈值。
另做自由截距 OLS：斜率 2623488.0000000005 B/B，截距 −1.12e−9 B，
R²=1，最大数值残差 4.66e−10 B（浮点求解误差）。

P4-T1 斜率是 3342336 B/B，斜率高估 718848 B/B（27.4005%）。
比例正确与绝对值准确是两项不同结果。这里存储尺寸由固定 shape 决定，
重复训练步不会产生存储尺寸方差；此三点验证的是本实现缓存布局在固定 S 下的比例关系，
不是对 RSS/traced peak 的线性拟合，也没有 held-out S、模型规模或并行度验证。

## 偏差分析

**常驻状态。** 实际参数与 P4 的精确参数计数一致，FP32 每份 1914880 B；
梯度也各分配一份。AdamW 只需 FP32 first/second moments，合计 3829760 B。
若直接沿用 P4-T1 默认 12 B optimizer，将得到 5744640 B，比本实现多 1914880 B
（相对本实现 +50%），原因是默认包含 master copy。此处通过已有 Precision 配置入口
选择 8 B，不改公式、不增设一个无实际作用的副本来消除偏差。
常驻表不包含 Python 对象头、优化器步数等标量或 allocator overhead。

**激活。** 手工反向在每层保存两个 norm 的 normalized/inverse_std、两个线性层输入、
QKV、softmax probability、attended、GeLU 前后值。按实际 inventory，在 B=1 时：

| 保存类别（两层合计） | 实测 B |
| --- | ---: |
| n1、n2、两个 normalized、QKV、attended、GeLU 前后（16×L×B×S×h 个 FP32） | 2097152 |
| 一个 attention probability（L×B×H×S² 个 FP32） | 524288 |
| 两个 norm 的 inverse_std（2×L×B×S 个 FP32） | 2048 |
| 合计 | 2623488 |

将这个缓存清单与原 P4 表达式代数对齐（f=4h、TP=PP=1）：
P4 是 `LBS[16ah+2kh+(2a+k)HS]`，当前缓存是 `LBS[16ah+aHS+2a]`。
这只是解释当前存储差额，并非提出或替换 P4 公式。B=1 的差额逐项为：

- P4 多估一份 FP32 attention 中间量：`LBS×aHS = 524288 B`。
- P4 预留 dropout masks，而 SUT dropout=0：`LBS×k(2h+HS) = 196608 B`。
- 本实现保存 norm inverse_std，原近似未单列：抵消 `LBS×2a = 2048 B`。

净值 `524288+196608−2048=718848 B`，与实际偏差完全一致，并随 B 缩放。
主导项是 attention 保留策略，其次是 dropout masks；这属于实现与保守激活近似的差异，
没有发现需要修正 P4-T1 公式的错误。全前向附加保留 393728×B B
（final norm、head input、logits）也说明 P4 的层内量不能直接等同端到端峰值。
反向临时数组、矩阵乘法 workspace、优化器临时 direction 还会影响 traced peak/RSS。

**步时。** 保留配置中的假想 c=1 TFLOP/s，不把它当本机标定值。
其 compute-only 时间对 B=1/2/4 分别为 0.417989/0.835977/1.671954 ms，
相对实测 p50 偏差为 −91.7248%/−89.1623%/−87.4103%。对应的 F/p50
工作量折算速率仅为 0.082752/0.108377/0.125897 TFLOP/s，不是硬件峰值或独立 FLOP 计数。
有效速率随 B 改变，因此没有用单点系数伪装整个范围的时间预测精度。
Python 调度、小矩阵利用率、norm/softmax/GeLU/CE、分配/释放和 tracing 扰动
都不在 P4 的简化 compute-only 模型中；本次未分别测量它们的因果贡献。
P0 [测量器标定](../p0/calibration.md)的 0.092584 ms 是特定探针的经验扰动尺度，
不能直接从当前步时扣除，也不支持声称这里存在某项优化收益。

## 边界声明

- **单机 CPU、单进程、DP=TP=PP=1，无多卡。通信量未实测，仅解析值**；
  当前配置的发送量解析值为 0 B/step，不能据此验证分布式实现。
- CPU ndarray 存储不是 GPU 显存测量。MPS/CUDA、ZeRO、张量/流水线并行、
  网络带宽、通信计算重叠、流水线气泡和重算均没有实测校准。
- P4-T1 的显存是假设理想均衡的常驻量与保留激活，时间是假想算力下的 compute-only 值。
  7B 扫描和多卡数据全部是解析模型，不是本机或真实多卡实验。
- 本次固定 S=128、478720 参数、FP32、无 dropout、无 checkpointing、synthetic tokens；
  未检验序列长度方向、训练精度、训练收敛或其他实现的缓存策略。
- 常驻状态逐项对齐不等于训练峰值内存预测正确；RSS 历史峰值和 tracemalloc 局部覆盖
  均有口径限制。三点存储线性不等于吞吐、总峰值或多卡内存线性。
- 测试与 `--check` 为流程/正确性 smoke，不计作正式校准证据；这份报告不修改路线图，
  阶段完成与合并由规划层独立验收决定。

## 复现与验收

正式配方：

```bash
uv run --locked python scripts/p4_calibrate.py
```

可用 `-o <路径>` 将完整校准表写入指定文件；每次都会另建带时间戳的三个 run，
不覆盖既有数据或自动改写本阶段报告。首次交付的 `-o` 表见
[calibration.md](../../runs/20261019-p4-calibrate-delivery/calibration.md)。
`--check` 保持 SUT 模型但改为 S=4、预热 0、重复 2，落盘明确标记为非实验证据。

测试对每个参数组做独立中心差分方向梯度检查，覆盖重复 token 的 embedding scatter-add
及 tied output 梯度；核对 SUT 前向一致、两步 AdamW 的 bias correction/解耦衰减、
缓存 alias 去重、正负及零分母偏差、线性斜率/截距/非线性识别、极小流程和 save/load hash。
测试不对机器速度设阈值。

规格第 5 节的验收命令为：

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/p4_calibrate.py --check
git status --short
```

实际验收结果与全部命令输出见
[验收日志](../../runs/20261019-p4-calibrate-delivery/acceptance.log)，
按规格第 6 节的文件清单、输出摘要、校准全表和偏差说明见
[-o 汇报](../../runs/20261019-p4-calibrate-delivery/task-report.md)。

# P4-T1 执行汇报 — 解析模型（非实测）

任务规格：tasks/p4-t1-models.md。分支：codex/p4-models。
已完成 DP/TP/PP、ZeRO-1/2/3、激活重算与梯度累积解析模型；128 个格网点经 save_run() 落盘并用 load_run() 回读校验。
本文件为任务第 6 节的汇报文件；最终答复可由调用方 codex exec 的 -o 参数另行保存。

1. 实际新增文件清单

- src/prism/parallel_models.py：公式、符号及单位定义、假设、输入校验和预测接口。
- tests/test_parallel_models.py：97 项测试，含独立公式校验、量纲缩放、守恒和记录回读。
- configs/p4_models.json：SUT 约 0.5M 与假想约 7B 模型、8 档并行策略、batch/累积/重算网格及精度。
- scripts/parallel_scan.py：全网格扫描、确定性 --check、save_run 记录、Markdown 表格及 -o/--output。
- runs/20261019-p4-models-095408665699/config.json
- runs/20261019-p4-models-095408665699/env.lock
- runs/20261019-p4-models-095408665699/measurement.json
- runs/20261019-p4-models-095408665699/meta.json
- runs/20261019-p4-models-095408665699/predictions.json
- runs/20261019-p4-models-095408665699/scan.md
- runs/20261019-p4-models-095408665699/acceptance.txt
- runs/20261019-p4-models-095408665699/report.md

全部新增路径均在白名单内；未修改现有模块、依赖、锁文件或测量协议；未删除文件，未 git push。runs 被现有 .gitignore 忽略，本次仅强制暂存上述白名单产物，未修改 .gitignore。

2. 第 5 节验收命令输出摘要

| 命令 | 最终输出摘要 | 退出码 |
| --- | --- | ---: |
| uv sync --locked | Resolved 9 packages；Checked 8 packages | 0 |
| uv run --locked pytest | 338 passed in 3.41s；其中新增 97 项 | 0 |
| uv run --locked ruff check . | All checks passed! | 0 |
| uv run --locked ruff format --check . | 42 files already formatted | 0 |
| uv run --locked python scripts/parallel_scan.py --check | 解析模型: --check passed, 128 grid points, no artifacts written | 0 |
| git status --short | 提交前仅白名单新增文件；提交后状态见最终答复 | 0 |

完整原始输出：acceptance.txt。另已运行正式扫描 uv run --locked python scripts/parallel_scan.py，成功保存本目录。
首次 lint 检出 B008（默认参数内构造 Precision）；改为函数内初始化后，以上最终整套验收全部通过。

3. 代表性格网点（全部为解析模型）

所有行 b=1、m=1、不重算；d/t/p/z 为 DP/TP/PP/ZeRO。存储精度：参数 2、梯度 2、Adam 12（含 FP32 master）、激活 2、mask 1 bytes/scalar。
显存单位 bytes/device，为理想均衡常驻状态与最坏 stage 保留激活之和；不是峰值显存。通信为 bytes/device/optimizer step，仅计发送；P2P 为 stage 均值。

| 模型 | d/t/p/z | 参数 B | 梯度 B | Adam B | 激活 B | 显存合计 B | AR B/step | AG B/step | RS B/step | P2P B/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prism-sut-0.5M | 1/1/1/0 | 957440 | 957440 | 5744640 | 1769472 | 9428992 | 0 | 0 | 0 | 0 |
| prism-sut-0.5M | 4/1/1/0 | 957440 | 957440 | 5744640 | 1769472 | 9428992 | 1436160 | 0 | 0 | 0 |
| prism-sut-0.5M | 4/1/1/1 | 957440 | 957440 | 1436160 | 1769472 | 5120512 | 0 | 718080 | 718080 | 0 |
| prism-sut-0.5M | 4/1/1/2 | 957440 | 239360 | 1436160 | 1769472 | 4402432 | 0 | 718080 | 718080 | 0 |
| prism-sut-0.5M | 4/1/1/3 | 239360 | 239360 | 1436160 | 1769472 | 3684352 | 0 | 1436160 | 718080 | 0 |
| prism-sut-0.5M | 1/2/1/0 | 478720 | 478720 | 2872320 | 1048576 | 4878336 | 262144 | 0 | 0 | 0 |
| prism-sut-0.5M | 1/1/2/0 | 478720 | 478720 | 2872320 | 884736 | 4714496 | 0 | 0 | 0 | 32768 |
| prism-sut-0.5M | 2/2/2/3 | 119680 | 119680 | 718080 | 524288 | 1481728 | 131072 | 239360 | 119680 | 32768 |
| hypothetical-7B | 1/1/1/0 | 13167247360 | 13167247360 | 79003484160 | 30601641984 | 135939620864 | 0 | 0 | 0 | 0 |
| hypothetical-7B | 4/1/1/2 | 13167247360 | 3291811840 | 19750871040 | 30601641984 | 66811572224 | 0 | 9875435520 | 9875435520 | 0 |
| hypothetical-7B | 4/1/1/3 | 3291811840 | 3291811840 | 19750871040 | 30601641984 | 56936136704 | 0 | 19750871040 | 9875435520 | 0 |
| hypothetical-7B | 2/2/2/3 | 1645905920 | 1645905920 | 9875435520 | 8321499136 | 21488746496 | 1073741824 | 3291811840 | 1645905920 | 16777216 |

SUT 参数量 478,720，已与 data/sut/kaleido-p3 实际 checkpoint 数组元素总数核对；假想 7B 参数量为 6,583,623,680，不对应真实模型权重。完整 128 行含重算、累积和静态吞吐见 scan.md 与 predictions.json。
宿主为 macOS 15.8 / arm64，16 CPU；不存在训练设备实测。静态吞吐是计算侧解析上界：假想每设备 1e12 FLOP/s，不对应任何真实硬件；未建模通信时间、通信–计算重叠、kernel 效率、调度延迟或实际负载不均。
预热 0、解析求值 1 次；measurement.json 的 p50=p95=p99 为第 0 格点的确定性预测，非实测采样分位数。内存字段 0 表示未测量，不能作为真实内存峰值。
扫描配方 SHA-256：130578147c31f929576fb0077d8b5a65f8bfefda68382cd3116952a58fce6e45。
save_run 完整配置 SHA-256：d09a9420f2df0b586962432a1efd2f8a574a9dd09212ed3df69774868588523e（包括记录兼容封装语义及参考行）。

公式抽查与守恒边界：

- ZeRO-2：q=P/(t*p)，参数保留 q*w，梯度与 Adam 状态分片为 q*g/d 与 q*o/d，总模型状态 q*[w+(g+o)/d] bytes；混合精度默认等于 q*(2+14/d)。量纲与分项测试覆盖所有 ZeRO 阶段。依据：[ZeRO，§3/§7](https://arxiv.org/abs/1910.02054)。
- PP bubble：stage 前后向时间 u 秒，有效时间 m*u，空闲时间 (p-1)*u，总时间 (m+p-1)*u，空闲/总时间为 (p-1)/(m+p-1)。论文的空闲/有效时间 (p-1)/m 经分母归一化即得到规格公式；测试验证 m=10^12 时趋零。依据：[Megatron，§2.2](https://arxiv.org/abs/2104.04473)。
- TP 激活：保留 10*S*b*h bytes 的复制项，不错误地将全部激活除以 TP；标准 FP16、f=4h 时复现论文式 (2)，另测任意 FFN 与字节宽度。依据：[Activation Recomputation，§4.2](https://arxiv.org/abs/2205.05198)。
- ZeRO-3 模型状态全集群守恒；包含激活的总显存守恒测试固定全局 microbatch、TP=PP=1、m=1，并分别覆盖重算开关。额外测试证明固定每设备 b 时，DP 激活复制令全集群总显存增加，避免声称 ZeRO 会自动分片激活。
- z>=2 每 microbatch 归约梯度以保持梯度分片常驻假设；z=3 每 microbatch 前后向各 gather 一次。全层重算将 TP 的每层 all-reduce 从 4 次增至 6 次。m=1、等精度时 ZeRO-1/2 通信量等于 DP，ZeRO-3 为 1.5 倍，已独立测试。

4. 与规格的偏差及原因

无。模型边界已在源码 docstring、配置及解析模型输出中说明：参数/状态假设理想均衡切分（含 embedding/bias/norm），激活是 1F1B 保留量近似；不含临时 all-gather、重算工作区、logits/loss、分配器碎片等峰值开销。SUT 激活使用带 dropout 的保守 GPT 近似，并非 SUT 训练实测。实测校准属于 P4-T2。

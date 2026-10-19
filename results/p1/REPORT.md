# P1 阶段验收报告

本报告收口 T1–T4 的代码、自测和实验记录，最终阶段验收仍由规划层独立重跑决定；未改动 ROADMAP 状态。
本机 CPU cached 回路相对 naive 约 1.48×；分页在本配方下增加内存且吞吐较低；校准后代表点中 continuous 的模拟优势保持，但绝对尾延迟明显变化。

## 交付物与证据

| 任务 | 交付物 | 合并/本次证据 |
| --- | --- | --- |
| T1 cache 回路 | `src/prism/kvcache.py`、`scripts/kvcache_bench.py`、`configs/p1_kvcache.json`、`tests/test_kvcache.py` | [PR #7](https://github.com/celanwang/prism/pull/7)，merge `22606f3`；[T1 comparison.json 所在归档](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-kvcache-093918473641-b1-naive) |
| T2 分页 | `src/prism/paged.py`、`scripts/paged_bench.py`、`configs/p1_paged.json`、`tests/test_paged.py` | [PR #9](https://github.com/celanwang/prism/pull/9)，merge `87b934e`；[T2 summary.json 所在归档](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-b8-m65536-allocation) |
| T3 调度模拟器 | `src/prism/schedsim.py`、`scripts/schedsim.py`、`configs/p1_schedsim.json`、`tests/test_schedsim.py` | [PR #6](https://github.com/celanwang/prism/pull/6)，merge `b0c81b5`；下表 before 为本次对原配置代表点的重跑 |
| T4 互校准与报告 | `scripts/p1_calibrate.py`、`configs/p1_calibrate.json`、`tests/test_p1_calibrate.py`、本报告 | 当前 `codex/p1-report` 分支（尚无 PR）；[calibration.json](../../runs/20261019-p1-calibrate-100535736151-prefill-1/calibration.json) |

T1 原始目录前缀 `runs/20261019-p1-kvcache-093918473641-*`；T2 使用后一次完整配方 `runs/20261019-p1-paged-095835826274-*`。
这些历史记录被 gitignore 忽略，本次在白名单 T4 runs 下按原始字节归档，未重新测量或改写其内容；[evidence_manifest.json](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence_manifest.json) 给出原路径、归档路径和逐文件 SHA-256。
其他轮次不与选定批次混算；顺序与运行漂移仍可能影响比较，不能选取最佳数字宣称分页普遍加速。

## 正确性门槛

T1 在计时前复用 SUT 完整性检查，并要求 cached 与 naive/参考 greedy token 逐个一致；logit 误差作为诊断，不另造精度门槛。
窗口内 fixture 共匹配 50 个 token（含 18 prompt），cache rebuild=0，最大绝对 logit 误差 9.5367431640625e-06；
跨窗 fixture 共匹配 143 个 token，rebuild=14，最大误差 6.198883056640625e-06。跨窗重编号后全量重建，不能复用旧绝对位置。
T2 的 continuous、paged block=8/16/32 均通过相同 fixture 与 rebuild 门槛，另有 54 个随机 prompt 分页对照（两个 seed × 九个边界长度 × 三个块长）全部一致。
本次 T4 运行先执行 `check_sut()`；极小测试只验证流程，正式数据来自完整配方。正确性证据见上述 T1/T2 归档摘要及现有独立测试。

## 加速比与内存

以下 T1/T2/T4 实测均为 Apple M4 Max / CPU / arm64 / 单进程、原生线程上限 1；SUT float32 小模型 478720 参数，2 层、4 heads、width=128、vocab=512、context=128。
环境指纹 `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；每个 run 的 env.lock 含 Python/NumPy/锁文件等完整记录。
T1 与 T2 generation 每 arm 预热 10、重复 200，包含 fresh-cache prompt prefill 与固定 32 个新 token；两种反序在同进程执行。p50/p95/p99 为样本 ms，NumPy linear 分位数，无置信区间。

| T1 顺序 | arm | p50 ms | p95 ms | p99 ms | traced peak bytes | RSS bytes | config SHA-256 | 原记录归档 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| naive→cached | naive | 24.805542000 | 25.471654150 | 27.475476750 | 621607 | 63930368 | `e203810a834f6b4e73f3b076167c8b393e66d8eb750cb4806d0fc1de4634f61a` | [20261019-p1-kvcache-093918473641-b1-naive](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-kvcache-093918473641-b1-naive) |
| naive→cached | cached | 16.753354000 | 17.045663950 | 17.350560740 | 261258 | 65683456 | `57df607f3734be14899266e6c0fc989ac4f08849ed551e6dea30772da84a8ddf` | [20261019-p1-kvcache-093918473641-b1-cached](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-kvcache-093918473641-b1-cached) |
| cached→naive | cached | 16.752625000 | 17.108546450 | 17.420034670 | 255546 | 65798144 | `19e962d2ed8ff449ca50eb32761f93fd86c11da7ae4e83487f7578a6fa7d4894` | [20261019-p1-kvcache-093918473641-b2-cached](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-kvcache-093918473641-b2-cached) |
| cached→naive | naive | 24.798229500 | 27.453253850 | 27.738010510 | 614743 | 66306048 | `2a6c56c8e461f1034a836977dba6d1d5b0a1c3f1d42fc672362d9a9a1d1df6d2` | [20261019-p1-kvcache-093918473641-b2-naive](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-kvcache-093918473641-b2-naive) |

两轮 naive/cached p50 加速比分别 1.480631×、1.480259×；p50 差 8.052188、8.045604 ms。
K/V 完整窗口解析值和 owning array 实际 nbytes 均为 262144 bytes；本配方最终 49 个已前向位置对应 100352 bytes。
cached−naive traced peak 差分别 −360349、−359197 bytes，但差值含 attention/logits/MLP 临时缓冲与 Python 簿记，不能当作 KV 大小。RSS 是 macOS 进程生命周期峰值，不能按 arm 相减解释独立占用；tracemalloc 也不覆盖全部原生分配。

## 分页内存–吞吐

横向比较固定 49 个已前向位置、相同生成 token；以下是离散内存–吞吐曲线。bytes/token=2048，连续 cache payload=100352 bytes。
P 是分页 block tokens；budget 只约束 owning K/V page arrays，不含权重、metadata、gather 与 forward 临时缓冲。fragmentation=(allocated−used)/allocated。

| P | budget bytes | generation | allocated bytes | fragmentation | 第1序 token/s | 第2序 token/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 65536 | infeasible | n/a | n/a | n/a | n/a |
| 8 | 114688 | feasible | 114688 | 0.125 | 1797.002005 | 1778.590940 |
| 8 | 131072 | feasible | 114688 | 0.125 | 1794.317095 | 1781.964603 |
| 16 | 65536 | infeasible | n/a | n/a | n/a | n/a |
| 16 | 114688 | infeasible | n/a | n/a | n/a | n/a |
| 16 | 131072 | feasible | 131072 | 0.234375 | 1849.953829 | 1846.000735 |
| 32 | 65536 | infeasible | n/a | n/a | n/a | n/a |
| 32 | 114688 | infeasible | n/a | n/a | n/a | n/a |
| 32 | 131072 | feasible | 131072 | 0.234375 | 1886.912999 | 1870.103270 |

65,536 bytes 的三个点及 P=16/32、budget=114,688 的两个点均不能保留完整正确窗口，因此没有 generation 时延或吞吐，不填零伪装测量。
可行分页占 114688 或 131072 bytes，相对连续 payload 增加 14.29% 或 30.61%；固定 T、嵌套块长下碎片由 12.5% 升至 23.4375%。
第1序 naive→continuous→paged，第2序 paged→continuous→naive。吞吐均按 32000/p50(ms) 推算，非独立均值或逐 token 分位数。下表给出曲线全部 generation 证据：

| 序 | arm / P / budget | p50 ms | p95 ms | p99 ms | token/s | config SHA-256 | 归档 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | naive | 27.220208500 | 28.244040200 | 29.166245750 | 1175.597167 | `75bb646e967f04ae9512b6f15626e4224eff6103377e9ca0e35acee13bfe8c99` | [20261019-p1-paged-095835826274-order1-naive-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-naive-decode) |
| 1 | continuous | 16.811042000 | 17.035539600 | 17.153835920 | 1903.510800 | `cc657382705c7c8d1facdda8d027e50d259e37ca43b3d1002db07d6e292c2a0c` | [20261019-p1-paged-095835826274-order1-continuous-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-continuous-decode) |
| 1 | paged / 8 / 114688 | 17.807437000 | 18.050006250 | 18.180870420 | 1797.002005 | `a2bc78678e69cb2dee94c079821dfb9586bc94a8c7c4947c80999e35c4dee39e` | [20261019-p1-paged-095835826274-order1-b8-m114688-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-b8-m114688-decode) |
| 1 | paged / 8 / 131072 | 17.834083000 | 18.073184800 | 18.138548240 | 1794.317095 | `b882671fa4180cfc7299f158f94d72c331a0deea06a388e2da3e2cbf3a63cab8` | [20261019-p1-paged-095835826274-order1-b8-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-b8-m131072-decode) |
| 1 | paged / 16 / 131072 | 17.297729000 | 17.454419050 | 17.546158660 | 1849.953829 | `52a8f8b88900ad0113f9183d4f1005d871aa93cc45e1bcdcbac7c39a659083f0` | [20261019-p1-paged-095835826274-order1-b16-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-b16-m131072-decode) |
| 1 | paged / 32 / 131072 | 16.958916500 | 17.211612500 | 18.608840590 | 1886.912999 | `e18843a76c49e77ee956d0dd060a69fda84a6aad3973eea7f281c91d1be0b9df` | [20261019-p1-paged-095835826274-order1-b32-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order1-b32-m131072-decode) |
| 2 | paged / 8 / 114688 | 17.991770500 | 18.291451750 | 18.672311010 | 1778.590940 | `ff8b5db9029b4f7f427de67e69f59bcc7f449a694a1c8bc9899565f2e2165a38` | [20261019-p1-paged-095835826274-order2-b8-m114688-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-b8-m114688-decode) |
| 2 | paged / 8 / 131072 | 17.957708000 | 18.208859800 | 18.242140920 | 1781.964603 | `7d4aa90dd0f22036e5254537134602fb02b400d80382335215835ad222c07d08` | [20261019-p1-paged-095835826274-order2-b8-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-b8-m131072-decode) |
| 2 | paged / 16 / 131072 | 17.334771000 | 17.614347900 | 18.027640500 | 1846.000735 | `22cdab7a8ed7b8bea50a3569915e52bbbc20e26f014523e25cf0d7bf126205a7` | [20261019-p1-paged-095835826274-order2-b16-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-b16-m131072-decode) |
| 2 | paged / 32 / 131072 | 17.111354500 | 17.417023550 | 17.624080000 | 1870.103270 | `e049705322341112504f599154483ed81f1a8e402f07238114a32981663e9763` | [20261019-p1-paged-095835826274-order2-b32-m131072-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-b32-m131072-decode) |
| 2 | continuous | 16.944750000 | 17.383953850 | 18.312299170 | 1888.490535 | `119e518266c55075aa69ad787d955b7f51d6dc30ee78d19cff5db8a5b9a8db54` | [20261019-p1-paged-095835826274-order2-continuous-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-continuous-decode) |
| 2 | naive | 25.103291500 | 28.191530900 | 28.356208660 | 1274.733236 | `662560d6b0aa7185bb897f74e83a220ef5c869e0592c7e155211a2dd83e5341d` | [20261019-p1-paged-095835826274-order2-naive-decode](../../runs/20261019-p1-calibrate-100535736151-prefill-1/evidence/20261019-p1-paged-095835826274-order2-naive-decode) |

选定完整批次中分页/连续 cache 的吞吐比第1序为 0.942636–0.991280，第2序为 0.941806–0.990264，全部低于 1。
这是分页负结果：该单序列 lazy-pages 配方既未节省 payload，也未提高吞吐；最小 p50 差 0.147875 ms，虽超过 P0-T4 的经验参照 0.092584 ms，仍不自动证明显著性或普遍劣化。
LRU/sliding 的命中、驱逐数字来自固定 synthetic trace 的内存模拟（预热10、重复200，见归档 summary 的 simulations）；它们不是真实注意力驱逐收益，也不能替代上述正确性门控后的生成实测。P1 未交付 KV 量化对照，不能据本报告宣称覆盖所有 KV 优化方向。

## 模拟–实测校准

T4 配方：prefill T∈{1,32,128}，token=42，使用 T1 `_forward_cached` 从空 cache 前向，输入构造在计时外，结果释放在计时内；预热100、重复1000。
decode B∈{1,2,4,8}，每序列用 token=(42+i)%512 构造独立 32-token cache，在计时外 prefill 并选取下一个 greedy 输入。
一个计时样本依次执行 B 次 batch-1 cached forward 和 greedy argmax，各序列从 32→33 个位置；以串行总耗时 ÷ 1 步计 decode(B)，**不除以 B**。
每样本复用未被修改的独立 past cache，丢弃新 cache/logits，因此无预热后持续增长、无窗口重建，也不在计时内重做 prefill；预热10、重复200。
这是 CPU 串行批成本实测代理，不存在并发或真实向量化 batch。全部实测经 measure()/save_run() 保存，保留 tracing 扰动。

| 点 | p50 ms | p95 ms | p99 ms | config SHA-256 | run |
| --- | ---: | ---: | ---: | --- | --- |
| prefill 1 | 0.491250000 | 0.549458050 | 0.603817670 | `ccf40cb17677fee31be70044e88cc2bad642ba00c4effb5d4fde2bad97327fc1` | [20261019-p1-calibrate-100535736151-prefill-1](../../runs/20261019-p1-calibrate-100535736151-prefill-1) |
| prefill 32 | 0.857542000 | 0.961338150 | 1.033101750 | `51f4aafce9b535ba6b8cba7f9101cd8dce2b8eea2a2848747853e83c24f9be91` | [20261019-p1-calibrate-100535736151-prefill-32](../../runs/20261019-p1-calibrate-100535736151-prefill-32) |
| prefill 128 | 1.705229500 | 1.914362800 | 2.105384830 | `3f337bad2ff0063fa8fb3a07dbfb98ef4519ed87af7f83d8f08b6cc7b219026f` | [20261019-p1-calibrate-100535736151-prefill-128](../../runs/20261019-p1-calibrate-100535736151-prefill-128) |
| decode 1 | 0.525750500 | 0.633021150 | 0.699656420 | `c62697e09b21a28659c12774d0d2ac5b8d48a26d706a2a2900bfb86f9aef3382` | [20261019-p1-calibrate-100535736151-decode-1](../../runs/20261019-p1-calibrate-100535736151-decode-1) |
| decode 2 | 1.029833500 | 1.106934800 | 1.155093830 | `6b874c7fcfc2305fd0a40dd44b14c5049e13fdfec3b33873337ccfdd1dd7a2b7` | [20261019-p1-calibrate-100535736151-decode-2](../../runs/20261019-p1-calibrate-100535736151-decode-2) |
| decode 4 | 2.062270500 | 2.173951450 | 2.278974170 | `344b8ce3a1ebf39abdd96230e4b6d4233e01bb7fc45612389359b903606e73dd` | [20261019-p1-calibrate-100535736151-decode-4](../../runs/20261019-p1-calibrate-100535736151-decode-4) |
| decode 8 | 4.163520500 | 4.401806250 | 4.639801160 | `6f2a4f1aad9b6334fa9e0af7a6ccb43311d804d28d5e129cc72c757a9fcd8d1e` | [20261019-p1-calibrate-100535736151-decode-8](../../runs/20261019-p1-calibrate-100535736151-decode-8) |

三点 OLS：prefill p50 = 0.514291155003 + 0.009386012640×T ms；拟合残差（实测−拟合）依次为 -0.032427168, +0.042898441, -0.010471273 ms。
模拟器仅支持 ms/token 斜率，本次只映射斜率，截距保留在 calibration.json 中供审查；未修改模拟协议去加入固定项。
原 T3 的 prefill=0.01224658203125 ms/token，decode(1)=0.8463639375 ms，由 P0 无 cache 回路外推；decode(B)=decode(1)×(1+0.35×(B−1))，0.35 为未校准假设。
本次校准副本完全替换服务模型节点为四个实测 p50，B=2 的原值由 T3 现有分段线性插值得到；未改动原 `configs/p1_schedsim.json`。副本 [calibrated_schedsim.json](../../runs/20261019-p1-calibrate-100535736151-prefill-1/calibrated_schedsim.json)；原配方快照 [original_schedsim.json](../../runs/20261019-p1-calibrate-100535736151-prefill-1/original_schedsim.json)。

## 偏差分析

相对偏差统一定义为 (模拟服务时间−实测 p50)/实测 p50；正值表示模拟更慢，负值表示低估。prefill 比较的是模拟器实际使用的 slope×T，未把拟合截距偷加回预测。

| 点 | 实测 p50 ms | 原模拟 ms | 校准模拟 ms | 原相对偏差 | 校准相对偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill 1 | 0.491250000 | 0.012246582 | 0.009386013 | -97.5071% | -98.0894% |
| prefill 32 | 0.857542000 | 0.391890625 | 0.300352404 | -54.3007% | -64.9752% |
| prefill 128 | 1.705229500 | 1.567562500 | 1.201409618 | -8.0732% | -29.5456% |
| decode 1 | 0.525750500 | 0.846363938 | 0.525750500 | +60.9821% | +0.0000% |
| decode 2 | 1.029833500 | 1.142591316 | 1.029833500 | +10.9491% | +0.0000% |
| decode 4 | 2.062270500 | 1.735046072 | 2.062270500 | -15.8672% | +0.0000% |
| decode 8 | 4.163520500 | 2.919955584 | 4.163520500 | -29.8681% | +0.0000% |

节点后的 decode 偏差为 0 是同点拟合的恒等结果，不是样本外准确性证据。decode 代理接近 B 线性增长，原模型对 B=1 偏高、对 B=4/8 偏低；
这混合了 P0 naive→T1 cached 路径差异和串行代理→解析批伸缩假设差异，不能把全部偏差归因于 0.35。
拟合得到正的固定截距，映射时省略它导致 prefill 短序列明显低估，三点的预测甚至比原模型更低；校准不是保证所有指标改善。
仅三个 prefill 点不足以描述注意力的非线性，单一 decode context=32 无法覆盖 T3 prompt=16–128 及生成后的上下文增长，更未模拟跨窗重建。

以下全部为**模拟**：同 seed=20261019、λ=100/s、B=4、到达窗口5000 ms，两策略 before/after 均 487 请求、排空后完成487，无拒绝/超时。
预热0、重复1（确定性事件模拟），模拟宿主环境仅作 provenance；E2E p50/p95/p99 取排空后的完成请求，token/s 仅计到达窗口内输出，排除 prompt。

| 服务模型 | 策略 | E2E p50 ms | p95 ms | p99 ms | 窗口 token/s | 最大队长 | 雪崩代理 | config SHA-256 | run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| before | static | 3493.459968 | 6725.465139 | 7068.234052 | 1421.000000 | 285 | True | `ec60034618acacd3762fc0de1ad3b15607981ccfd81fae1dc85634556975440f` | [20261019-p1-calibrate-100535736151-before-static](../../runs/20261019-p1-calibrate-100535736151-before-static) |
| before | continuous | 1448.007500 | 2893.622673 | 3013.698028 | 2172.000000 | 178 | True | `f32f2a6a46fa9cc10b8e793cabef925420754a1bb55464348acb1421d05281ea` | [20261019-p1-calibrate-100535736151-before-continuous](../../runs/20261019-p1-calibrate-100535736151-before-continuous) |
| after | static | 4403.071241 | 8618.826965 | 9127.939348 | 1242.600000 | 312 | True | `da28e1cf266a8393cbca2a7cfc0d46f5ff33d7c2dadf81eca1fedd570389f33d` | [20261019-p1-calibrate-100535736151-after-static](../../runs/20261019-p1-calibrate-100535736151-after-static) |
| after | continuous | 2095.329264 | 4155.357832 | 4323.149065 | 1864.600000 | 220 | True | `a6370bd39bf81b1e4a027159db0ece0580d998829d00ce8010d511928ca5cb31` | [20261019-p1-calibrate-100535736151-after-continuous](../../runs/20261019-p1-calibrate-100535736151-after-continuous) |

代表点 continuous/static 模拟窗口吞吐比 1.528501→1.500563，continuous 的 p50/p99 和队长也更低，定性优势未翻转。
但校准后 static p99 增加 29.14%、continuous p99 增加 43.45%；服务模型误差会放大为排队尾延迟误差。
四个记录均同时触发 p99>1000 ms 与有限窗口队长增长代理；所以 λ=100/s、B=4 这一点的雪崩判定未改变。
**雪崩点位置是否移动尚不能判定**：本规格只重跑一个 λ、B 代表点，没有校准后 λ 扫描。不能把该点“仍雪崩”表述成临界到达率或整个网格边界不变；有限窗口代理也不是无限队列发散的证明。

## 边界声明

结论限于本机 CPU、单线程、小模型、固定输入和 tracemalloc 协议。未测 MPS、CUDA、多卡、HTTP 服务或线上并发。
批成本来自同进程 B 个独立 batch-1 序列的串行总耗时代理；没有真实 batch 的 kernel 融合、权重共享访存或并行效率，不能外推 GPU batching 收益。
continuous batching 的 E2E、吞吐与雪崩均是调度模拟结论；只有服务时间代理是实测，尚无端到端服务实测互校验。
样本来自单轮顺序测量，无随机交叉、多进程独立重复或置信区间；P0-T4 的 0.092584 ms 只是经验扰动参照，不能由增加 repeat 消除单样本 tracing 成本。
节点内零残差、smoke 流程通过不等于实验预测有效。P1 的 KV 量化未在 T1–T4 中实现；阶段收口范围和是否放行由规划层决定。

## 复现与验收

正式配方：`uv run --locked python scripts/p1_calibrate.py`。每次生成新的 `runs/<UTC日期>-p1-calibrate-*`，保存全部样本、环境、源文件哈希、拟合与偏差、两份模拟配置、事件/请求/队列与前后汇总；不覆盖本报告。
小配方：`uv run --locked python scripts/p1_calibrate.py --check`（warmup=0/repeat=2、模拟窗口最多500 ms，持久化并标注 smoke，不作为本报告证据）。

固定校准节点重放全部四点（不重新测量）：

```bash
uv run --locked python scripts/p1_calibrate.py --replay runs/20261019-p1-calibrate-100535736151-prefill-1/calibration.json
```

重放会逐项比较新旧完整模拟 summary（含 config hash），不一致即失败。

第5节全部验收命令：

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/p1_calibrate.py --check
git status --short
```

本次执行：`uv sync --locked` 成功；`pytest` 460 passed（6.19s）；`ruff check` 全绿；`ruff format --check` 50 files already formatted；`--check` 完成7个实测代理点与4个模拟点（仅 smoke）；`git status --short` 已核对白名单。正式节点重放4个模拟 summary 全部一致。

执行结果亦在本次 `-o /tmp/p1-t4-report.md` 汇报中列出；最终验收由规划层按任务规格独立重跑。

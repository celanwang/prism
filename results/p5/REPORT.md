# P5：独立进程复跑与 v0.1.0 发布候选审阅

建议 **v0.1.0 作为完整仓库 clone 的研究 CLI 候选**。P0–P4 的冻结证据校验通过，
同机干净 clone、新 venv 的完整代表配方复跑通过；性能漂移如实保留，
不以速度相等为放行条件。执行层收口完成；规划层的独立验收、合并与实际发布仍待进行。
本任务不打 tag、不 push、不修改版本或历史证据。

## 证据链总览

| 阶段 | 正式报告与证据入口 | 已合并 PR | 审阅结论与范围 |
| --- | --- | --- | --- |
| P0 | [基准闭环](../p0/REPORT.md)、[T4 标定](../p0/calibration.md)、SUT MANIFEST、baseline runs | [#1](https://github.com/celanwang/prism/pull/1)、[#2](https://github.com/celanwang/prism/pull/2)、[#3](https://github.com/celanwang/prism/pull/3)、[#4](https://github.com/celanwang/prism/pull/4)、[#5](https://github.com/celanwang/prism/pull/5) | 测量协议、计时原始样本、配置/环境哈希完整；T1 单独为本地验收 merge `86fd087`，没有 PR，不能将 #1 算作 T1 |
| P1 | [KV/分页/调度校准](../p1/REPORT.md)，其中链接原字节 evidence 归档与 manifest | [#6](https://github.com/celanwang/prism/pull/6)、[#7](https://github.com/celanwang/prism/pull/7)、[#9](https://github.com/celanwang/prism/pull/9)、[#10](https://github.com/celanwang/prism/pull/10) | 真实 KV cache/分页回路与调度模拟分列；KV cache 量化没有实现，不能把阶段完成解读为愿景中所有子题均完成 |
| P2 | [量化](../p2/quant.md)、[投机解码](../p2/speculative.md)、[算子消融](../p2/kernels.md) | [#11](https://github.com/celanwang/prism/pull/11)、[#14](https://github.com/celanwang/prism/pull/14)、[#16](https://github.com/celanwang/prism/pull/16) | 分项报告而非不存在的 REPORT.md；精度损失、反量化成本与算子负结果均保留 |
| P3 | [服务/负载与雪崩](../p3/REPORT.md)，服务等价性与负载归档在对应 runs | [#12](https://github.com/celanwang/prism/pull/12)、[#15](https://github.com/celanwang/prism/pull/15) | loopback HTTP/SSE 为实测；理想队列预测与实测的偏差明确，不作为生产容量保证 |
| P4 | [解析模型及真实单机训练步校准](../p4/REPORT.md) | [#8](https://github.com/celanwang/prism/pull/8)、[#13](https://github.com/celanwang/prism/pull/13) | 常驻数组存储对齐与激活比例成立；绝对激活量存在偏差，多卡通信仅解析模型 |
| P5 | [复现清单](../../configs/reproduce.json)、[冻结 inventory](../../artifacts/inventory-v1.json)、本报告与下列独立复跑记录 | [#17](https://github.com/celanwang/prism/pull/17)；P5-T2 当前分支待验收 | 复用已合并 CLI，无测量协议或依赖修改 |

PR 指针均与 clone 中 `git log` 的 merge 记录对应。`prism verify` 校验
**1591** 个冻结文件，`source_commit=124fe7c4ef5c655655cfb11012f1fd1fac184910`；
该提交是 inventory 的历史锚点，不是本次候选提交。历史报告、配置、代码和关键 runs
保持原字节；P5 的新增文件不在旧 inventory 中，由候选快照、发布记录和本次 git diff 追踪。

## 独立复跑

首轮完整证据：[release.json](../../runs/20261019-p5-release-125418964850-full/release.json)、
[完整比对](../../runs/20261019-p5-release-125418964850-full/reproduction.json)、
[原始记录及逐配方日志归档](../../runs/20261019-p5-release-125418964850-full/reproduction.tar.gz)。
源基线为 rebase 到 PR #17 后的 `3a72da1e7b609359c568e4d07d117f4f7091bfac`，
临时候选提交为 `dfb918fa5f288ad5de16dc35583e81d9eee271d2`。
[candidate.patch](../../runs/20261019-p5-release-125418964850-full/candidate.patch) 与 `candidate/`
保存当时六个白名单输入；补丁 SHA-256：
`193f10df25abdaffbfea82af39901619aad3a931cd48280192c8d87276b48e0b`。

流程为临时 seed clone → 仅白名单候选输入提交 → 第二次干净 clone →
`uv sync --locked` 新建 `.venv` → 全量 pytest → verify → README 命令 → 完整抽样 →
clone 状态复核。Python/uv/git/BLAS 的继承覆盖变量清除，未复制原 venv；
uv 的下载缓存可以复用。临时目录结束前归档日志和原始结果，不依赖已消失的绝对路径。
归档中 `reproduction/<会话>/<配方>/runs/` 对应比对 JSON 的相对 run 路径。

运行于 **Apple M4 Max / CPU / arm64 / 16 逻辑 CPU / 48 GiB RAM**，
macOS 15.8、Python 3.12.15、NumPy 2.5.4；各配方在独立进程执行并设原生线程上限 1。
采集区间为 2026-10-19 12:54:18–12:56:19 UTC。
所有 67 次历史锚点比对的配置哈希相等，环境指纹均为
`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`，本机环境差异数 0；
代码与测试也覆盖 `env_equal=false` 的显式报告路径，不把环境不等静默解释为一致。

P5-T1 抽样为 baseline、p1-calibrate、quant、serve、p4-calibrate，含依赖共 **10** 个配方。
共 **93** 个标准 run：**67** 次有锚点比对（同一个历史 before 锚点可在不同配方中重复比对）、
**16** 个 unanchored 调度网格点、**10** 个 new-derived 新拟合模拟结果。
其中 5 次冻结模拟/解析产物比对完全相等，62 次实测只要求配置一致并报告性能漂移。
T4 自身没有标准 run，只比对报告中的配置/环境指纹；本次均相等，未冒充历史报告逐字节重建。

| 配方 | 历史锚点比对数 | env 差异 | 超过 T4 的 p50/p95/p99 漂移计数 | 额外结果 |
| --- | ---: | ---: | --- | --- |
| sut / calibrate | 0 / 0 | 不适用 | 不适用 | SUT 正确性通过；T4 报告指纹相等 |
| baseline | 4 | 0 | 1 / 1 / 1 | 无缺失 |
| kvcache | 4 | 0 | 3 / 3 / 3 | 无缺失 |
| schedsim | 2 | 0 | 不适用（模拟） | 16 unanchored；两个冻结点确定性相等 |
| p1-calibrate | 9 | 0 | 0 / 3 / 5 | 两个 before 确定性相等；2 new-derived after |
| quant | 36 | 0 | 12 / 15 / 15 | 无缺失 |
| serve | 8 | 0 | 8 / 8 / 8 | 8 new-derived 模拟结果 |
| parallel | 1 | 0 | 不适用（解析） | predictions.json 相等 |
| p4-calibrate | 3 | 0 | 3 / 2 / 2 | 无缺失 |

合计 p50/p95/p99 分别有 27/32/34 次漂移超过 **0.092584 ms**。
标注使用冻结的 P0-T4 经验扰动尺度；它不是显著性检验、速度合格线或跨环境保证，
没有因为漂移而改历史记录或重新拟合历史输入，也没有据此宣称历史优化收益稳定。
完整 signed drift 与每个 run 的 config hash 在 reproduction.json；原始分位数、samples、
预热/重复、配置与 env.lock 在归档中，不能将不同工作负载的漂移计数解释成时延分布。

测量协议原样保留：T4 预热100/重复2000；baseline prefill 100/1000、decode 10/200；
kvcache 10/200；P1 校准 prefill 100/1000、decode 10/200；quant 3/30 且两个相反顺序批次；
serve 2/5 个完整轮次（请求分位数与轮次分位数分别记录）；P4 校准 1 次存储诊断、5 次预热、30 次测量。
每项 p50/p95/p99 使用既有 P0 协议；模拟没有实测计时分布。
抽样配置文件 SHA-256 为 `055ca68a289463af9914b9dd5532c0ba38bc84befaad88deca7025eef9920ad0`；
发布清单 SHA-256 为 `df7a6a9b02d06dcd96200e580a059b92b4d22a22c73d925346be08c9c52a77e0`。
这些是文件字节哈希；每个测量的 canonical config hash 是另一口径，已逐项与历史相等核验。

首轮 clone 中：pytest **750 passed**，ruff check 全绿，format **87 files already formatted**；
README 11 条 shell 命令均实际执行成功，`--all --dry-run` 只核验全量计划，不宣称全量配方都重测。
clone 最后 `git status --short` 为空。

负结果保留：[首次尝试](../../runs/20261019-p5-release-125334130717-full/release.json)
在文档核对前因 macOS `/var` 为软链接而被安全路径检查拒绝，尚未运行正式抽样。
修复为规范化临时目录路径后才得到上述通过结果。随后增加归档排除既有发布目录和证据文件哈希，
并测试防止未来 clone 把旧发布归档再次嵌套复制；最终复跑和规格验收记录见文末。

## 文档逐项核对与差异清单

| 文档 | 核对结果 | 处理与保留边界 |
| --- | --- | --- |
| README | 两个 shell 代码块共 11 条命令全部通过；包含 sync、pytest、两条 ruff、模块入口、两个 verify、list、全量 dry-run、完整 sample、release --check | 仅追加复现与验证指引；旧的“P0 尚未开始”及“runs 不入库”未擅自改写，追加说明明确它们的历史性与已入库例外 |
| ROADMAP | P0/P1/P3/P4 的 REPORT 与 P2 三个分项报告均存在并受 inventory 覆盖；各阶段 PR 与合并历史对应 | 只改 P5 状态与证据指针为“候选审阅完成，待规划层独立验收与合并”；未修改 P0–P4 状态或扩大验收范围 |
| DESIGN | kaleido-p3 工件已冻结入库；运行无需开发者的 kaleido 工作目录；实际 CPU 证据与无 CUDA/多卡实测的边界相符 | 设计愿景是探索范围，不代表所有子题都已交付；MPS 是允许范围，本次没有 MPS 实验 |
| AGENTS | 规格→codex 分支执行→diff/独立验收的流程一致；白名单、locked 依赖、禁 push/删文件/改历史均遵守 | P5-T2 为低风险独立验证工具任务；本脚本的独立进程不是 Kimi 的独立验收，不替代规划层最终合并决定 |
| 历史 results | P0/P1 尚无 PR、P2 草稿、P4 待验收等措辞是当时的交付状态；现在已合并的指针见本报告总览 | 历史报告受 inventory 保护，原样保留；未为文案一致而改历史哈希。P1 未实现 KV 量化等范围差异显式保留 |
| P5 inventory 边界 | 冻结清单覆盖 P0–P4 和关键输入，不覆盖自身或新增 P5 记录 | 本任务白名单不包含 inventory；P5 候选输入用 patch/哈希，新增证据用发布摘要哈希和 git 追踪，不声称旧 inventory 覆盖全仓每一个文件 |

机械检查只证明报告路径、ROADMAP 指针、合并 PR、冻结覆盖和命令可执行性；
边界与结论的语义核对由执行层逐项审阅并在本节说明，不能仅凭文档中存在“完成”二字认定全部愿景达成。

## 已知边界与免责声明汇总

- 实测限同一台 Apple Silicon CPU、小型 FP32 NumPy SUT；本次新 clone/新进程只增加同机复跑证据，
  不代表跨硬件、跨 OS、第三方复现或置信区间。没有 MPS/CUDA/多卡实测，模拟不表述为实测。
- P0 tracemalloc 只覆盖增量分配且漏掉部分 native 内存；RSS 是进程生命周期高水位。
  经验 T4 不是可扣除的固定计时成本，增加 repeat 不能消除单样本扰动。
- P1 分页与 continuous batching 的适用范围见报告；调度预测依赖服务时间代理，
  未用本次新拟合覆盖历史模型。unanchored/new-derived 不计为历史一致；KV 量化仍为空缺。
- P2 压缩载荷不等于峰值内存或速度收益；INT4 有输出/精度损失，INT8 可因反量化而变慢。
  投机实现校验 FP32 target 的 greedy 逐 token 等价，未验证随机采样分布；
  算子消融的可分辨性和顺序漂移限制按原报告保留，本次代表抽样未重测所有算子。
- P3 是 loopback、无 TLS、单进程串行推理；请求线程不等于并行推理。
  成功请求分位数不含拒绝/丢弃；短窗、有界队列的“雪崩”代理不是无限队列发散证明，
  原模拟存在低估尾时延的负结果，不能做生产容量保证；本次抽样不是重跑整个负载扫描。
- P4 DP/TP/PP/ZeRO 通信及时间为理想解析假设；真实校准只有单机 DP=TP=PP=1，
  ndarray 存储不等于 GPU 显存，常驻存储对齐和激活比例正确不等于绝对峰值/时间预测正确。
- CLI 发布范围是完整仓库 clone。wheel 缺少 scripts/configs/inventory/SUT，不能独立做全仓复现。
  smoke/test 只证明流程或正确性；本报告的实验证据来自完整 sample，未把 `--check` 当正式实验。

## 版本建议与收口

v0.1.0 的研究 CLI 候选证据成立：冻结工件完整、完整代表配方配置一致、
确定性模拟可重建、性能及环境差异有明确记录、README 可执行、阶段证据与状态对应。
不以仍未探索的愿景子题阻止本次已批准范围收口；这些空缺与负结果均已显式说明。
最终发布与 P5 合并由规划层按 AGENTS 独立验收决定。本任务没有执行发布动作。


## 最终脚本复跑及规格验收

最终脚本复跑在 2026-10-19 12:59:18–13:01:25 UTC 完成：
[release.json](../../runs/20261019-p5-release-125918660239-full/release.json)、
[逐记录比对](../../runs/20261019-p5-release-125918660239-full/reproduction.json)、
[原始产物](../../runs/20261019-p5-release-125918660239-full/reproduction.tar.gz)。
该轮候选快照为 `d0a872d41299a5fe66912ebedec402e214ceda49`，
树为 `2964eb62976f202ed809ec58a5695555acbd4294`；代码、测试、配置、README、ROADMAP
与最终交付相同，本报告在测量后追加本节，输入版本保存在该轮 candidate/ 与 patch 中。

该轮全量测试 **751 passed（34.76s）**，两条 ruff 全绿（87 files）；
README 11 条命令再次全部通过，1591 个冻结文件校验通过，clone 结束状态为空。
仍为 10 个配方、93 个 run、67 次配置及环境相等、5 次冻结确定性产物相等；
16 unanchored / 10 new-derived 未被计为历史一致。
最终轮 T4 超尺度计数如下，与首轮的区别是如实记录的性能漂移，不是配置变化：

| 实测配方 | 超 T4 的 p50 / p95 / p99 计数 |
| --- | --- |
| baseline | 1 / 1 / 1 |
| kvcache | 4 / 4 / 4 |
| p1-calibrate | 3 / 5 / 5 |
| quant | 15 / 20 / 24 |
| serve | 8 / 8 / 8 |
| p4-calibrate | 3 / 3 / 3 |
| 合计 | 34 / 41 / 45 |

硬件、设备、预热/重复、分位数口径和冻结配置均与上节相同；各测量 config hash
在最终轮 reproduction.json 中逐项列出。未从两次复跑中选择速度更好的一次。
最终轮 `evidence_sha256` 覆盖 69 个归档文件；原始压缩归档 SHA-256 为
`7db1d9232ef22671a44c9e670616cba86c5f619722e3ffa6f1da0204a35fbb0d`。
[归档复核结果](../../runs/20261019-p5-release-delivery/artifact-validation.json)
确认两轮分别解包重载 93 个原始 run 后，按原 P5-T1 比对函数重新计算得到的全部记录与原摘要完全一致；
最终轮全部 69 个证据哈希也重新核验相等。

规格第 5 节的最终执行摘要与逐命令输出写入
[验收记录](../../runs/20261019-p5-release-delivery/acceptance.json)，
规格第 6 节完整交付汇报写入
[task-report.md](../../runs/20261019-p5-release-delivery/task-report.md)。
验收日志是生成产物；本报告不宣称测量前的快照已经包含事后生成的结果。

最终规格验收全部返回 0：sync 解析9包/检查8包；pytest **751 passed（37.79s）**；
ruff check 全绿；format 87 files；release `--check` 通过（[小规模记录](../../runs/20261019-p5-release-130339649116-check/release.json)）；
提交前 status 仅列出白名单源文件。`--check` 不作为正式测量证据。
与任务规格的偏差：**无**。初次临时目录路径失败与修复、历史文档差异、
P5-T1 抽样的原有覆盖限制均如实记录；没有越界改写历史证据来获得通过。

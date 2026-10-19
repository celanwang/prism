# P2-T1 执行汇报（-o）

任务规格：`tasks/p2-t1-quant.md`；当前分支：`codex/p2-quant`。
已按规划层批准的实施计划与六项边界完成实现；待规划层独立验收，未宣称阶段完成。

## 1. 实际新增文件

本次提交共 152 个新增文件，原有文件零修改。

- `src/prism/quant.py`：INT8 每输出通道、INT4 按输入列分组真打包，fp32 scales，ties-to-even；
  QuantizedSUT 仅覆盖 Linear，复用原始 forward/generate 与 cached 回路；不持有源模型或 fp32 Linear 副本。
- `tests/test_quant.py`：29 项参数化测试，覆盖 INT8 半 scale 误差界、INT4 独立标量解包、
  零组、尾组、奇数 nibble、极值/次正规数、非法参数、fp32 bit-exact 不变、缓存 oracle、
  payload 簿记、teacher-forced argmax 定位、非有限失败及 --check 不落盘。
- `configs/p2_quant.json`：单线程 CPU、INT4 group_size=32、两种相反方案顺序、预热 3/重复 30；
  embedding 与共享输出头保留 fp32，quantize_embedding=true 明确报错。
- `scripts/quant_bench.py`：完整精度对照、prefill/isolated cached decode、实际 Linear 输入的成分重放探针；
  正式运行 save_run 后 load_run 校验，--check 仅校验与测量并打印 JSON。
- `results/p2/quant.md`：三角数据报告草稿；`results/p2/acceptance.json`：全部验收 stdout/stderr/退出码；
  `results/p2/execution-report.md`：本文件。
- 36 个正式 `runs/20261019-p2-quant-101317066291-*` 目录，逐条 4 文件，首条另有 comparison.json。

未修改 sut.py、kvcache.py、measure.py、env.py、record.py、任何原有测试或依赖文件；未 push、未删除文件。
初轮探索测量与回归命令部分重叠，因此正式表使用随后独立重跑的数据。初轮的
`runs/20261019-p2-quant-101249856947-*` 仅本地保留，未纳入证据或本次提交。

## 2. 规格第 5 节验收输出摘要

| 命令 | 退出码 | 输出摘要 |
| --- | --- | --- |
| `uv sync --locked` | 0 | Resolved 9 packages；Checked 8 packages |
| `uv run --locked pytest` | 0 | 477 passed in 7.04s（原有 448 项 + 新增 29 项） |
| `uv run --locked ruff check .` | 0 | All checks passed! |
| `uv run --locked ruff format --check .` | 0 | 50 files already formatted |
| `uv run --locked prism sut check` | 0 | passed=true；2 greedy cases、32 argmax positions、3 tokenizer prompts |
| `uv run --locked python scripts/quant_bench.py --check` | 0 | implementation_passed=true；完整 36 条测量；accuracy_gate=null |
| `git status --short` | 0 | 提交前仅白名单新增文件，提交后再次核对 |

完整输出保存于 acceptance.json。以 SHA256 快照确认 --check 前后 runs/ 与 results/ 文件内容不变。
全部 36 条正式记录经 load_run 校验，sources_sha256 与最终源码一致。
INT8 round-trip 以 abs(dequantize(quantize(W))-W) ≤ scale/2 + float32 舍入容差断言；
独立测试同时验证构造量化模型前后源 arrays、fp32 forward 输出完全一致。
首次自测暴露并修复了接近 float32 最大值时的 scale 向上舍入溢出；最终极值测试通过。

## 3. 三角数据全表与可复现锚点

硬件 Apple M4 Max / arm64 / macOS 15.8 / 16 逻辑 CPU / 48 GiB；设备 CPU、BLAS 单线程。
所有探针使用原始 measure() 协议：预热 3 次、重复 30 次，保留原始 samples，报告 p50/p95/p99。
环境指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。
配方 JSON 哈希（sorted JSON）：`227ea07e25cb03497e29c365b624dee433bf29100678f4c49e659b9b97e30968`；每条完整测量 config hash 见下表。
比较汇总：`runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/comparison.json`（含逐层权重误差、逐随机用例、teacher-forced 定位）。

观察：INT8/INT4 的持久数组载荷分别降至 744448/587776 bytes，整模型压缩比分别约 2.57×/3.26×。
两批完整回路中量化方案的 prefill/decode p50 均高于 fp32；此实现没有观察到时延收益。
本次 INT4 的 fixture 与随机生成出现明显退化；不设精度合格阈值，不能把实现通过写成精度可接受。
cached/非 cached 如有微浮点 argmax 差异，仅在相同 teacher-forced 前缀定位；本次正式用例未观察到该类分歧。
内存结论只基于精确载荷簿记；峰值包含其他临时缓冲且各模型共存，只作旁证。

# P2-T1 权重量化报告草稿（待规划层终审）

CPU 实测：Apple M4 Max；设备 CPU；单线程。
预热 3 次，重复 30 次；两个相反顺序批次；单位 ms。
实现通过不表示精度可接受；本任务不设精度合格阈值。

## 精度与精确载荷簿记

token 一致率只计生成后缀；首分歧为生成后缀零基索引。logits 对比 reference_logits.npz。

| 方案 | within 一致率/首分歧 | rollover 一致率/首分歧 | logits max/MSE | 随机一致率 | 模型 bytes | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 1.000000/None | 1.000000/None | 9.53674316e-06/1.94212968e-12 | 1.000000 | 1914880 | 1.000000× |
| int8 | 1.000000/None | 1.000000/None | 0.0651092529/0.000232012259 | 1.000000 | 744448 | 2.572215× |
| int4 | 0.968750/24 | 0.562500/9 | 0.57213974/0.0251380081 | 0.562500 | 587776 | 3.257840× |

随机 prompts：种子 42，8 个，长度 12，生成 16；对比 cached fp32。

精确簿记（包含 padding 和 fp32 scales，零点为隐式 0；不计 Python 元数据）：

```json
{
  "fp32": {
    "total_bytes": 1914880,
    "compression_ratio": 1.0
  },
  "int8": {
    "fp32_baseline_bytes": 1914880,
    "linear_fp32_bytes": 1572864,
    "retained_fp32_bytes": 342016,
    "integer_bytes_including_padding": 393216,
    "scale_bytes": 9216,
    "zero_point_bytes": 0,
    "total_bytes": 744448,
    "compression_ratio": 2.5722145804676755,
    "linear_compression_ratio": 3.9083969465648853
  },
  "int4": {
    "fp32_baseline_bytes": 1914880,
    "linear_fp32_bytes": 1572864,
    "retained_fp32_bytes": 342016,
    "integer_bytes_including_padding": 196608,
    "scale_bytes": 49152,
    "zero_point_bytes": 0,
    "total_bytes": 587776,
    "compression_ratio": 3.2578397212543555,
    "linear_compression_ratio": 6.4
  }
}
```

## 时延与峰值旁证

prefill 为整段提示；decode 为固定 fp32 前缀下 32 个 cached 步骤，排除 prefill。
每条记录含完整 config hash、env 指纹及原始 samples。

| batch | 方案 | 阶段 | 成分 | p50 | p95 | p99 | traced peak bytes | RSS bytes | config hash | env 指纹 | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fp32 | prefill | full | 0.793667 | 0.881550 | 0.949235 | 251728 | 86130688 | 0376384da6c2830e3e4a49145bcf8bf60cdddb094bd8eee5b33b4bcb2e2eb337 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full |
| 1 | fp32 | prefill | dequantize | 0.000417 | 0.000505 | 0.000601 | 392 | 87572480 | 0885ab7f1625ff40426efdf4767c41a41602f33e351261f65c0178125fed1c19 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-prefill-dequantize |
| 1 | fp32 | prefill | matmul_bias | 0.106896 | 0.109975 | 0.119819 | 117560 | 87605248 | 0f1d22a99df935dd3d24aed70a7d470660c51b50e04aaef678abd02366ba1bfa | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-prefill-matmul_bias |
| 1 | fp32 | decode | full | 16.597854 | 16.879200 | 16.967694 | 221846 | 88489984 | eb0eaa73c6bd5ac9fa7958b9e8ad8a6382208a826978190ea4243617954d8f25 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-decode-full |
| 1 | fp32 | decode | dequantize | 0.000417 | 0.000541 | 0.000630 | 392 | 88522752 | 05d18de827722fcf7d5f00a480f019a5b6942bdf3f8e2eb864b8aaef2e981c7d | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-decode-dequantize |
| 1 | fp32 | decode | matmul_bias | 1.103937 | 1.111277 | 1.114697 | 6456 | 88539136 | edd5fa04cabae094454f86c5b598d2b4ad07f7dd03438189697386cf73b4fe59 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-fp32-decode-matmul_bias |
| 1 | int8 | prefill | full | 1.024417 | 1.153431 | 1.230305 | 701595 | 88571904 | f8c57153af0bc07985fdb281f79e4e1e36a01f037f27c6ac19fe69e4e1616bec | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-prefill-full |
| 1 | int8 | prefill | dequantize | 0.196666 | 0.215592 | 0.217353 | 558760 | 88588288 | ade1e4afcc51853dfe5b81784afec2c9d4c0968399342972dbd9c835a34673c1 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-prefill-dequantize |
| 1 | int8 | prefill | matmul_bias | 0.153875 | 0.159979 | 0.160581 | 117560 | 88588288 | 2a756a3204e5333dc5cd7fe513d86c33e3c4b2b72a6ea901e3d11cae2bfdeecb | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-prefill-matmul_bias |
| 1 | int8 | decode | full | 24.095500 | 24.421525 | 24.471276 | 774149 | 89079808 | 34a235933814979b96c8e1f8d5000aab152ade7f509ec780b7d3285f8a993152 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-decode-full |
| 1 | int8 | decode | dequantize | 6.146021 | 6.653915 | 6.710978 | 558760 | 89112576 | c7ed08d69b9b3a1b2057b9bcc80d69f2ac95b82f10287d08cdee911a5facda58 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-decode-dequantize |
| 1 | int8 | decode | matmul_bias | 1.097979 | 1.128600 | 1.155396 | 6456 | 89112576 | da33af3321331b15a425fd60c2f9fea5025b5d51d632f0c182401031800c633c | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int8-decode-matmul_bias |
| 1 | int4 | prefill | full | 3.490437 | 3.526512 | 3.535648 | 1061063 | 89194496 | 8dd095c7fb4f23a0f330602e574235b24001a4c174c5bb57474f4db1c7cfbeff | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-prefill-full |
| 1 | int4 | prefill | dequantize | 2.691959 | 2.930504 | 3.009977 | 918576 | 89210880 | 3cdc2df97c23922246b50f9c04a47c29fb7d7a18900cdab81078b547180b4afc | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-prefill-dequantize |
| 1 | int4 | prefill | matmul_bias | 0.130000 | 0.144489 | 0.145314 | 117560 | 89210880 | 45d8be1b161d16ae0786a8b7208e7e32dedff709788927a027bc7cdf189ad8b7 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-prefill-matmul_bias |
| 1 | int4 | decode | full | 105.977979 | 113.830477 | 115.781459 | 1134076 | 94371840 | 0793bf9c0be34fc47d43e40b12dd1e6823a19c2384fc7e5a173ecd6926de68f1 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-decode-full |
| 1 | int4 | decode | dequantize | 90.002229 | 98.144921 | 98.508511 | 918576 | 94470144 | 4b25de89f8f76b3a0c81416f0cf234ef09cca928148df9b60b43849205c2c3b0 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-decode-dequantize |
| 1 | int4 | decode | matmul_bias | 1.130812 | 1.163042 | 1.179086 | 6456 | 94470144 | 94881c9fef1fccac04a454d42ad9ad632052c4a92c99c561c4ac55a6590842e3 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b1-int4-decode-matmul_bias |
| 2 | int4 | prefill | full | 3.671250 | 3.782873 | 3.794437 | 1061114 | 94470144 | ec67bfc029d16b556007659b2fd6e956d3c500c5e4f35f2b2b5916a380f5bcce | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-prefill-full |
| 2 | int4 | prefill | dequantize | 2.997521 | 3.342064 | 3.432294 | 918576 | 94486528 | 5a2f64fe8cec57785432151ca8a35e2798d665d63f60b753e24d827b6b943c60 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-prefill-dequantize |
| 2 | int4 | prefill | matmul_bias | 0.169000 | 0.182958 | 0.183224 | 117560 | 94519296 | 6009a4559cfa2b78ab89c201c1060ecacbc263126e50cd286a0f6774cc3ee6c4 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-prefill-matmul_bias |
| 2 | int4 | decode | full | 109.604833 | 114.456562 | 115.971764 | 1133872 | 94978048 | d517d58bdde8f1ef80569499bea3444c340dd09846988b0dec0d615c8066f39e | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-decode-full |
| 2 | int4 | decode | dequantize | 88.995854 | 96.685920 | 100.318743 | 918576 | 95043584 | f9dc8be8751404ab5ccf21ff0c4a23d2e27f84b29b606d23d40b8ea02902b749 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-decode-dequantize |
| 2 | int4 | decode | matmul_bias | 1.127104 | 1.222729 | 1.236670 | 6456 | 95043584 | b5c6ccbdae592e7f69fb4ea266f2ea89ea43844a435a9c823ac86abb2ae743d3 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int4-decode-matmul_bias |
| 2 | int8 | prefill | full | 1.069729 | 1.251227 | 1.372183 | 701544 | 96829440 | 62ff91fe7c563bdfca49c53961e166d37b3b6b5c57a09cd326c38780b08073a9 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-prefill-full |
| 2 | int8 | prefill | dequantize | 0.155917 | 0.172815 | 0.175850 | 558760 | 96845824 | 4cc73777f7a22a25859165a2c6bb8af89bbf72f7207e78a90e8febbf8f795ede | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-prefill-dequantize |
| 2 | int8 | prefill | matmul_bias | 0.145792 | 0.177944 | 0.187142 | 117560 | 96845824 | a0f3d9171a81b3c0e189ff87e73d35f2d0f68d930014547ba05e753265d9da64 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-prefill-matmul_bias |
| 2 | int8 | decode | full | 24.297812 | 27.387379 | 30.484484 | 774149 | 98385920 | 245d434e3458af7e88d662339e09962dcf521fe6ae9be308055f32fe5a7bd282 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-decode-full |
| 2 | int8 | decode | dequantize | 6.775729 | 7.190471 | 7.234548 | 558760 | 98385920 | 933ce9b043d2f2492130b5039ca661ed29db6352efd35db2eae1b1c0b22d0b72 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-decode-dequantize |
| 2 | int8 | decode | matmul_bias | 1.111125 | 1.166020 | 1.187092 | 6456 | 98385920 | a37c533bad3f1f5bccf9b84a355ec1e10b5fb22ee27d053f3c3642d037262502 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-int8-decode-matmul_bias |
| 2 | fp32 | prefill | full | 0.827291 | 0.886258 | 0.926990 | 251236 | 98385920 | 4bb35a2a2571a1ff1c1884f59bbd6e489b9e64ade8a4fa6a5b75adc60fb0cf34 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-prefill-full |
| 2 | fp32 | prefill | dequantize | 0.000417 | 0.000546 | 0.000643 | 392 | 98385920 | be7960cdbc841521e6bdd944e85e2215f05b5a3e562c043d2961b56bc9f432a5 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-prefill-dequantize |
| 2 | fp32 | prefill | matmul_bias | 0.153896 | 0.160781 | 0.173034 | 117560 | 98385920 | e3c9f5334f8209c33812c26f200d33d5be4a5a2dd87d9a646a4bcff36f2f6b74 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-prefill-matmul_bias |
| 2 | fp32 | decode | full | 16.579292 | 16.903575 | 17.126200 | 221693 | 98861056 | 791930d21446cd6d0d5f8797ae91406c7f53b9c5330732204f5bf8de7427ab5c | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-decode-full |
| 2 | fp32 | decode | dequantize | 0.000459 | 0.000584 | 0.000702 | 392 | 98861056 | 67703d1e20df995c4ce986c9cfd17b7975d8c2564e7e242db406da1b4b8178d5 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-decode-dequantize |
| 2 | fp32 | decode | matmul_bias | 1.109625 | 1.128235 | 1.130867 | 6456 | 98861056 | 15de8993c10ff0e03aca4045e034b707f9ad309de8034f0603fc32ddf3ae0a27 | bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3 | runs/20261019-p2-quant-101317066291-b2-fp32-decode-matmul_bias |

## 反量化与矩阵乘占比（独立探针估算）

来自实际 Linear 输入的重放；分母为完整回路 mean。独立测量并不保证相加为 100%，matmul 含 bias，输出头仍属 fp32 且不计入此 Linear 探针。

| batch | 方案 | 阶段 | 反量化/full mean | matmul+bias/full mean |
| --- | --- | --- | --- | --- |
| 1 | fp32 | prefill | 0 (无反量化) | 13.23% |
| 1 | fp32 | decode | 0 (无反量化) | 6.64% |
| 1 | int8 | prefill | 19.07% | 14.83% |
| 1 | int8 | decode | 26.06% | 4.57% |
| 1 | int4 | prefill | 78.34% | 3.80% |
| 1 | int4 | decode | 84.37% | 1.06% |
| 2 | int4 | prefill | 82.88% | 4.62% |
| 2 | int4 | decode | 82.01% | 1.03% |
| 2 | int8 | prefill | 15.35% | 14.31% |
| 2 | int8 | decode | 27.62% | 4.54% |
| 2 | fp32 | prefill | 0 (无反量化) | 18.64% |
| 2 | fp32 | decode | 0 (无反量化) | 6.69% |

## T4 经验阈值

阈值 0.092584 ms，来源 results/p0/calibration.md；低于该阈值的差异不作为收益证据。超过阈值不自动证明显著性。

```json
[
  {
    "batch": 1,
    "phase": "prefill",
    "scheme": "int8",
    "fp32_minus_quant_p50_ms": -0.23074950000000005,
    "p50_speedup": 0.7747503090783875,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 1,
    "phase": "prefill",
    "scheme": "int4",
    "fp32_minus_quant_p50_ms": -2.6967704999999995,
    "p50_speedup": 0.22738324350457503,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 1,
    "phase": "decode",
    "scheme": "int8",
    "fp32_minus_quant_p50_ms": -7.497646000000003,
    "p50_speedup": 0.688836255732398,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 1,
    "phase": "decode",
    "scheme": "int4",
    "fp32_minus_quant_p50_ms": -89.380125,
    "p50_speedup": 0.1566160645505421,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 2,
    "phase": "prefill",
    "scheme": "int8",
    "fp32_minus_quant_p50_ms": -0.24243700000000012,
    "p50_speedup": 0.7733658587202266,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 2,
    "phase": "prefill",
    "scheme": "int4",
    "fp32_minus_quant_p50_ms": -2.8439585000000003,
    "p50_speedup": 0.2253432754511406,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 2,
    "phase": "decode",
    "scheme": "int8",
    "fp32_minus_quant_p50_ms": -7.718520999999999,
    "p50_speedup": 0.6823367947217471,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  },
  {
    "batch": 2,
    "phase": "decode",
    "scheme": "int4",
    "fp32_minus_quant_p50_ms": -93.0255415,
    "p50_speedup": 0.1512642375906909,
    "T4": "超过 T4 经验参照；不自动证明显著性"
  }
]
```

## 实现、诊断和边界

0.5M small model, CPU, fp32 activations/KV/embeddings/tied head; weight-only. Dequantize on every Linear, then fp32 matmul: no native integer MAC acceleration. No accuracy acceptance threshold. Implementation passed does not mean acceptable accuracy. Decode uses a fixed fp32 teacher-forced continuation and immutable prefill cache, no rollover; full decoding includes K/V concatenations. Components replay captured actual Linear inputs; dequantize and matmul+bias are separately measured probes, not additive causal attribution. Persistent payload ledger excludes Python metadata and transient buffers; tied head counted once. All comparison models and probe buffers coexist; tracemalloc peaks are incremental, miss native allocations; RSS is process lifetime high-water mark, neither proves model compression. Counterbalanced single-process orders, no confidence intervals. T4 floor is a historical heuristic.

INT8 每输出通道 [-127,127]；INT4 按输入列分组 [-7,7]，真 nibble 打包，低 nibble 在先；fp32 scales，ties-to-even，尾组补 0。embedding 与共享头保留 fp32。
逐层权重误差、随机逐用例结果、cached/非 cached 的 teacher-forced 前缀定位见首个 run 的 comparison.json。

复现：`uv run --locked python scripts/quant_bench.py`；`--check` 校验并执行完整配方，仅打印 JSON，不写盘。

## 4. 与规格的偏差及原因

无。应用用户已批准的六项边界；公共接口与测量协议保持原样。
组件占比来自实际 Linear 激活的独立重放测量，不能解释为精确可加的因果分解；
decode 以相同 fp32 teacher-forced token 测 32 个 cached 步骤并排除 prefill，避免量化贪心分歧改变负载。
限制均已在配方/报告中声明：小模型、CPU、仅权重量化、每次 Linear 先反量化再 fp32 矩阵乘；
没有原生 INT MAC 加速结论；没有显著性或泛化精度结论。

## 提交文件完整清单

```text
src/prism/quant.py
tests/test_quant.py
configs/p2_quant.json
scripts/quant_bench.py
results/p2/quant.md
results/p2/acceptance.json
results/p2/execution-report.md
runs/20261019-p2-quant-101317066291-b1-fp32-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-full/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/comparison.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-fp32-prefill-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-full/config.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-int4-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-int4-prefill-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-full/config.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b1-int8-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b1-int8-prefill-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-full/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-fp32-prefill-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-full/config.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-int4-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-int4-prefill-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-decode-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-full/config.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-full/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-decode-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-full/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-decode-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-decode-matmul_bias/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-dequantize/config.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-dequantize/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-prefill-dequantize/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-dequantize/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-full/config.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-full/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-prefill-full/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-full/meta.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-matmul_bias/config.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-matmul_bias/env.lock
runs/20261019-p2-quant-101317066291-b2-int8-prefill-matmul_bias/measurement.json
runs/20261019-p2-quant-101317066291-b2-int8-prefill-matmul_bias/meta.json
```

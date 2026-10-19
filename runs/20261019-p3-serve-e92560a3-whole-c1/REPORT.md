# P3-T1 HTTP 服务与部署等价性

硬件/设备：Apple M4 Max / CPU / 1 native thread；每个 grid point 预热 0 轮、重复 2 轮。
smoke 非实验证据。
分位数均用 NumPy linear；请求级只统计成功请求（含排队），拒绝单独记录。
Measurement 是完整一轮耗时，不能解读为单请求时延。

硬门槛：fixture within_window/window_rollover + 固定种子随机 prompt，
HTTP 整包 == SSE done/逐 token 重建 == 独立本地 generate(use_cache=True)。
完整序列见 equivalence.json；事件门控真流式验证见 tests/test_serve.py。

| 模式 | C | 成功/请求 | 429 | 请求 E2E p50/p95/p99 ms | 等待 p50/p95/p99 ms | 整轮 p50/p95/p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 实测 whole | 1 | 16/16 | 0 | 9.060/9.578/9.734 | 0.029/0.033/0.037 | 68.075/68.244/68.259 |
| 模拟 whole | 1 | 14/24 | 10 | 23.249/27.615/27.796 | 11.375/20.504/20.653 | 无 |
| 实测 whole | 2 | 16/16 | 0 | 12.038/18.607/18.609 | 2.808/8.729/8.736 | 67.555/67.597/67.601 |
| 模拟 whole | 2 | 14/24 | 10 | 23.454/32.489/33.672 | 13.172/20.544/20.957 | 无 |
| 实测 whole | 4 | 12/16 | 4 | 17.149/25.703/25.816 | 7.595/16.157/17.285 | 50.848/51.269/51.306 |
| 模拟 whole | 4 | 11/24 | 13 | 30.787/32.567/32.645 | 18.208/23.779/25.216 | 无 |
| 实测 whole | 8 | 6/16 | 10 | 20.478/28.452/29.348 | 9.714/18.852/19.244 | 28.310/30.271/30.445 |
| 模拟 whole | 8 | 7/24 | 17 | 24.482/32.361/32.929 | 14.471/23.213/23.367 | 无 |
| 实测 stream | 1 | 16/16 | 0 | 10.548/11.094/11.115 | 0.068/0.167/0.169 | 78.125/78.151/78.154 |
| 模拟 stream | 1 | 16/24 | 8 | 21.592/26.244/28.808 | 10.618/16.364/17.695 | 无 |
| 实测 stream | 2 | 16/16 | 0 | 14.046/21.469/21.619 | 3.561/10.441/10.482 | 78.060/78.294/78.315 |
| 模拟 stream | 2 | 16/24 | 8 | 21.664/26.308/28.914 | 10.724/16.465/17.758 | 无 |
| 实测 stream | 4 | 12/16 | 4 | 21.572/31.640/32.959 | 11.110/21.232/21.356 | 61.532/63.464/63.636 |
| 模拟 stream | 4 | 12/24 | 12 | 27.139/33.437/34.605 | 17.172/21.048/21.196 | 无 |
| 实测 stream | 8 | 6/16 | 10 | 26.105/34.809/35.389 | 13.071/22.992/23.214 | 34.713/35.641/35.723 |
| 模拟 stream | 8 | 8/24 | 16 | 29.132/32.929/33.086 | 17.100/23.231/23.761 | 无 |

T3 同名义参数：batch_size=1、batch_timeout_ms=0、相同 Q 与长度分布。
模拟的 offered rate 为实测每轮请求数/整轮时长；模拟 Poisson 到达，实测固定序列分波闭环。
T3 保留 P0 无缓存服务成本代理（来源、哈希、近似均在模拟 config.json）；
未标定 HTTP/cached 成本，未建模 socket/线程/客户端与 tracemalloc 开销，
长度与服务耗时相关性也不同。仅做分布对照，不比较逐请求轨迹或声称拟合精度。
高并发拒绝数受 OS/线程调度影响；固定种子固定的是请求序列和并发模式。
等待仅含服务推理队列；TTFT、拒绝 E2E 和每请求明细见 summary.json/requests.json。

- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c1`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `afd5424107b21053eea682d3cea65e3b05a41e8cdfd79c0682015c2e053171c3`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c1-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c2`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `c503b488d19d9139899e4610c6291feb5dce11afbb0c10015797e56d28364396`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c2-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c4`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `8355f97f08207ddb751163a6e9ba24b105030ee7acc68f212e1dc9ba93aadfaa`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c4-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c8`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `1d4341e4410c9dbdc96ff8be1de45a356b5dbe65e581b627314e577d92793462`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-whole-c8-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c1`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `2bf9c18dbd00cd8087aaf0c4da3392fcdb0331e64d12947f6b034999ce566eea`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c1-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c2`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `5b796aa909a8e756a78006a6493b1e0af890deb424109b78aa7b907eaa78922a`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c2-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c4`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `66cd313c65d13bccccb6fbc432db5931220c2f7503aac14ded8a3c898ef169fe`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c4-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c8`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `1997062222daf16bdd8ab9cb92fd9c774787884cd3ba04f027dc64b65fecbef1`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-e92560a3-stream-c8-sim`。

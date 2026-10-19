# P3-T1 HTTP 服务与部署等价性

硬件/设备：Apple M4 Max / CPU / 1 native thread；每个 grid point 预热 2 轮、重复 5 轮。
实测：本机 loopback、同进程客户端与服务、小模型；不代表生产性能。
分位数均用 NumPy linear；请求级只统计成功请求（含排队），拒绝单独记录。
Measurement 是完整一轮耗时，不能解读为单请求时延。

硬门槛：fixture within_window/window_rollover + 固定种子随机 prompt，
HTTP 整包 == SSE done/逐 token 重建 == 独立本地 generate(use_cache=True)。
完整序列见 equivalence.json；事件门控真流式验证见 tests/test_serve.py。

| 模式 | C | 成功/请求 | 429 | 请求 E2E p50/p95/p99 ms | 等待 p50/p95/p99 ms | 整轮 p50/p95/p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 实测 whole | 1 | 160/160 | 0 | 8.083/9.321/9.465 | 0.024/0.031/0.037 | 250.073/250.902/251.045 |
| 模拟 whole | 1 | 112/172 | 60 | 23.830/33.138/34.933 | 13.485/21.934/23.643 | 无 |
| 实测 whole | 2 | 160/160 | 0 | 11.940/18.149/18.872 | 2.542/8.657/8.777 | 252.288/254.502/254.827 |
| 模拟 whole | 2 | 114/172 | 58 | 24.910/34.047/36.363 | 13.848/23.278/24.401 | 无 |
| 实测 whole | 4 | 120/160 | 40 | 16.985/25.401/27.273 | 7.662/16.792/17.624 | 189.730/190.076/190.110 |
| 模拟 whole | 4 | 91/172 | 81 | 25.379/33.563/36.854 | 14.518/23.634/24.375 | 无 |
| 实测 whole | 8 | 60/160 | 100 | 18.759/27.947/30.127 | 9.317/18.824/19.510 | 104.328/111.746/112.377 |
| 模拟 whole | 8 | 53/172 | 119 | 27.493/34.037/34.766 | 17.364/23.142/24.801 | 无 |
| 实测 stream | 1 | 160/160 | 0 | 9.760/11.611/11.885 | 0.067/0.204/0.222 | 304.782/309.511/309.585 |
| 模拟 stream | 1 | 128/172 | 44 | 21.617/32.010/35.737 | 11.060/21.860/24.914 | 无 |
| 实测 stream | 2 | 160/160 | 0 | 14.578/21.555/23.120 | 3.115/10.548/10.920 | 300.478/309.786/311.218 |
| 模拟 stream | 2 | 127/172 | 45 | 21.600/32.010/36.010 | 10.898/21.402/24.718 | 无 |
| 实测 stream | 4 | 118/160 | 42 | 20.395/31.457/33.044 | 9.317/19.949/21.618 | 224.506/235.471/236.641 |
| 模拟 stream | 4 | 104/172 | 68 | 25.028/33.660/37.683 | 14.130/22.825/25.708 | 无 |
| 实测 stream | 8 | 60/160 | 100 | 22.416/33.834/34.365 | 11.976/21.460/22.502 | 124.338/129.961/130.357 |
| 模拟 stream | 8 | 60/172 | 112 | 29.362/36.076/38.445 | 18.410/25.865/26.970 | 无 |

T3 同名义参数：batch_size=1、batch_timeout_ms=0、相同 Q 与长度分布。
模拟的 offered rate 为实测每轮请求数/整轮时长；模拟 Poisson 到达，实测固定序列分波闭环。
T3 保留 P0 无缓存服务成本代理（来源、哈希、近似均在模拟 config.json）；
未标定 HTTP/cached 成本，未建模 socket/线程/客户端与 tracemalloc 开销，
长度与服务耗时相关性也不同。仅做分布对照，不比较逐请求轨迹或声称拟合精度。
高并发拒绝数受 OS/线程调度影响；固定种子固定的是请求序列和并发模式。
等待仅含服务推理队列；TTFT、拒绝 E2E 和每请求明细见 summary.json/requests.json。

- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c1`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `7e01398714811cbedafe57b36e762da04ff7dcb2a8c781fd9c9d76461b0ef040`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c1-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c2`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `aa61597cba9c65370217bae35637340c35704d7981dbbb0d871cad7db6252e42`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c2-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c4`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `59ea707dce255fff770f99b1fbeda94ed7a03f47bcde86f10466949df979778a`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c4-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c8`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `9ee73d31efb3beb286ae4e0ae79fc03429b25621f5d2ed699859aa4f944de352`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-whole-c8-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c1`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `719017239c5ce4d69dd11e15916c1fbcdf503789956c593e8ab97255d408126e`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c1-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c2`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `ccc372d814c382a5f2a7a79e29739683b39bf18c9517a7bffbfbecd5e204a8ba`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c2-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c4`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `a41a41c2b330c741da6f5ef5f80ed9ecb26fc6b22b357a53dc311c80188e5214`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c4-sim`。
- `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c8`
  env `bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`；config `70ad66ee73bef2f80471da3e26ac57fe89c2660176fff028fade599ec1c5f2f6`；
  模拟 `/Users/wangchen/Developer/prism-wt-serve/runs/20261019-p3-serve-138d0748-stream-c8-sim`。

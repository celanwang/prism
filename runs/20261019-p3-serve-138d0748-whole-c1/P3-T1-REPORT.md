# P3-T1 执行汇报（tasks/p3-t1-serve.md 第 6 节）

分支：`codex/p3-serve`。本文件为本次 -o 汇报交付文件。

## 1. 实际新增文件

- `src/prism/serve.py`：仅绑定 127.0.0.1；有界 HTTP 工作线程池（含 pending 上限）、单推理 worker/FIFO 队列；429/503 分离；JSON 与 SSE；输入和输出限制；错误响应；启停与资源回收。线程模型在模块 docstring。
- `tests/test_serve.py`：24 项新增用例（含参数化实例）。独立 cached generate 为 oracle，覆盖 fixture 两用例和固定种子随机 prompt；初始裁剪、多次窗口重建、字符串/Unicode、EOS/零输出、事件门控真流式、并发隔离/FIFO、429/503、错误恢复、断连/取消、端口释放。只断言行为性质，15 秒为测试死锁诊断预算，不是性能门槛。
- `scripts/serve_bench.py`：固定种子波次压测、整包/SSE × 并发 1/2/4/8；P0 measure()/save_run()；独立等价性门槛；T3 同名义参数分布对照；--check 与 -o/--output。
- `configs/p3_serve.json`：CPU 单 native thread、Q=2、HTTP workers=16、pending=0；固定种子 20261019，完整模式每轮 32 请求、预热 2 轮、重复 5 轮。--check 每轮 8 请求、预热 0、重复 2，标注 smoke 非实验证据。
- 下列 runs 目录包括 8 组实测和 8 组模拟的完整记录，另有独立 --check 产生的同样 16 份 smoke 记录。全部 32 份均以 load_run() 成功校验 config 哈希、环境指纹、Measurement 和元数据。完整实际文件清单见同目录 `files.txt`。

- `runs/20261019-p3-serve-138d0748-stream-c1/`
- `runs/20261019-p3-serve-138d0748-stream-c1-sim/`
- `runs/20261019-p3-serve-138d0748-stream-c2/`
- `runs/20261019-p3-serve-138d0748-stream-c2-sim/`
- `runs/20261019-p3-serve-138d0748-stream-c4/`
- `runs/20261019-p3-serve-138d0748-stream-c4-sim/`
- `runs/20261019-p3-serve-138d0748-stream-c8/`
- `runs/20261019-p3-serve-138d0748-stream-c8-sim/`
- `runs/20261019-p3-serve-138d0748-whole-c1/`
- `runs/20261019-p3-serve-138d0748-whole-c1-sim/`
- `runs/20261019-p3-serve-138d0748-whole-c2/`
- `runs/20261019-p3-serve-138d0748-whole-c2-sim/`
- `runs/20261019-p3-serve-138d0748-whole-c4/`
- `runs/20261019-p3-serve-138d0748-whole-c4-sim/`
- `runs/20261019-p3-serve-138d0748-whole-c8/`
- `runs/20261019-p3-serve-138d0748-whole-c8-sim/`
- `runs/20261019-p3-serve-e92560a3-stream-c1/`
- `runs/20261019-p3-serve-e92560a3-stream-c1-sim/`
- `runs/20261019-p3-serve-e92560a3-stream-c2/`
- `runs/20261019-p3-serve-e92560a3-stream-c2-sim/`
- `runs/20261019-p3-serve-e92560a3-stream-c4/`
- `runs/20261019-p3-serve-e92560a3-stream-c4-sim/`
- `runs/20261019-p3-serve-e92560a3-stream-c8/`
- `runs/20261019-p3-serve-e92560a3-stream-c8-sim/`
- `runs/20261019-p3-serve-e92560a3-whole-c1/`
- `runs/20261019-p3-serve-e92560a3-whole-c1-sim/`
- `runs/20261019-p3-serve-e92560a3-whole-c2/`
- `runs/20261019-p3-serve-e92560a3-whole-c2-sim/`
- `runs/20261019-p3-serve-e92560a3-whole-c4/`
- `runs/20261019-p3-serve-e92560a3-whole-c4-sim/`
- `runs/20261019-p3-serve-e92560a3-whole-c8/`
- `runs/20261019-p3-serve-e92560a3-whole-c8-sim/`

## 2. 第 5 节验收命令

| 命令 | 退出码 | 输出摘要 |
| --- | --- | --- |
| `uv sync --locked` | 0 | Resolved 9 packages；Checked 8 packages |
| `uv run --locked pytest` | 0 | 484 passed in 10.02s |
| `uv run --locked ruff check .` | 0 | All checks passed! |
| `uv run --locked ruff format --check .` | 0 | 54 files already formatted |
| `uv run --locked prism sut check` | 0 | passed=true；2 greedy cases、32 logits argmax positions、3 tokenizer prompts |
| `uv run --locked python scripts/serve_bench.py --check` | 0 | 6/6 等价性用例，整包/SSE × 4 并发，本命令单独写入 16 个 save_run 目录（8 实测 + 8 模拟） |
| `git status --short` | 0 | 提交前仅白名单新增文件；完整输出见 git-status-before-commit.txt；提交后工作区干净 |

注：全流程（完整模式 + 独立 --check）共 32 个 save_run 目录，单次 --check 为 16 个。
验收摘要与 SUT 原始输出见 acceptance.json，--check 原始输出见 check-output.json。
另以独立服务进程运行 curl：整包 == SSE 重建 == 本地 cached generate；4 个 token 事件，meta/done 完整；SIGINT 干净退出 0。原始响应为 curl-generate.json / curl-stream.txt（文本末尾空行规范化）。真实逐 token 到达由事件门控测试验收，不以 curl 的累计输出推断。

## 3. 部署等价性与并发压测证据

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

本机实测现象：固定 Q=2 时，C=1/2 无拒绝，C=4/8 背压生效；整包 C=1 到 C=8 的成功请求 E2E p99 从约 9.465ms 到 30.127ms，SSE 从约 11.885ms 到 34.365ms。这些数字只对应上表硬件/预热/重复/配置哈希；分位数未给出置信区间，拒绝和调度仍受 OS 影响。高并发时整轮耗时下降伴随大量 429，不能据此声称成功请求吞吐提高。SSE 首 token 数据和拒绝响应耗时保存在 summary.json；原始每请求明细在 requests.json。

T3 与 HTTP 的拒绝数、E2E/等待分布存在明显差异。除了到达过程不同，T3 沿用 P0 无缓存代理，不是当前 cached HTTP 成本模型；这里如实保留差异，不将分布接近或偏差解读为校准通过。

## 4. 与规格偏差

无。按已批准裁决实现；未修改现有 src/prism 模块、README/results、依赖或测量协议；未 git push。任务消息未提供具体 -o 路径，依照“报告写 runs”将此文件作为汇报交付路径。

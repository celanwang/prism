# 任务规格 P3-T1：HTTP 并发推理服务与部署等价性

- 阶段：P3 服务与部署（docs/ROADMAP.md）
- 分支：`codex/p3-serve`
- 风险等级：**高**（引入并发与网络边界，等价性门槛必须严防死锁/竞态）
- 门控流程：双阶段——先只读输出实施计划，规划层批准后才放行执行。
- 依赖：P1-T1（cached 回路）；P2 不阻塞本任务

## 1. 研究问题

"并发与流式如何改变尾时延"的工程半边：
把 prism 的 cached 推理回路包成 HTTP 服务后，
**服务输出与本地推理是否逐 token 等价**（部署等价性）？
并发请求在单进程串行推理下的排队行为与模拟器预测差多少？

可能的失效形式：

- 服务路径与本地路径数值分叉（浮点环境、线程、tokenizer 差异）
  （对策：部署等价性硬门槛——同 prompt 服务输出与本地
  generate 逐 token 一致）。
- 并发测试 flaky（对策：确定性压测——固定种子、固定并发模式、
  断言行为类性质而非绝对时延）。
- 依赖膨胀（对策：仅标准库 http.server + threading，不许框架）。

## 2. 改动白名单

- 新增 `src/prism/serve.py`
- 新增 `tests/test_serve.py`
- 新增 `configs/p3_serve.json`
- 新增 `scripts/serve_bench.py`
- 新增 `runs/<日期>-p3-serve-*/` 产物目录

## 3. 禁止事项

- 修改 `src/prism/` 现有模块；引入 Web 框架或任何新依赖
- 改依赖、`git push`、删除文件；实现鉴权/TLS（本阶段范围外，
  服务仅绑定 127.0.0.1）
- 把压测数字表述为生产级性能结论

## 4. 实现要求

1. **HTTP 服务**（标准库 http.server + 线程池，绑定 127.0.0.1，
   端口可配）：
   - `POST /generate`：{prompt, max_new_tokens} → {token_ids, ...}
   - `POST /generate/stream`：SSE 或 chunked 逐 token 流式返回
   - `GET /healthz`；请求体大小限制；JSON 错误响应
   - 推理串行化：单推理 worker + 请求队列（队列长度可配，
     满了返回 429 背压），工作线程模型写入文档
2. **部署等价性（硬门槛）**：服务输出的 token_ids 与本地
   `sut.generate(..., use_cache=True)` 对同一 prompt 逐 token 一致；
   覆盖 fixture 两用例 + 随机 prompt；流式与整包输出一致。
3. **并发压测**：`configs/p3_serve.json` 配方——固定种子生成
   请求序列，并发度 {1,2,4,8}，记录每请求等待/端到端时延、
   429 计数、p50/p95/p99，经 save_run() 落盘；
   与 T3 模拟器同参数对照（标注实测 vs 模拟）。
4. **测试**：等价性门槛、429 背压、流式完整性、错误路径、
   服务启停干净（端口释放）；并发测试只断言行为性质
   （全部完成、顺序性、背压生效），不断言绝对时延。
5. 风格：ruff line-length 100、全类型注解、docstring 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked prism sut check
uv run --locked python scripts/serve_bench.py --check
git status --short
```

人工检查项：规划层独立启动服务，用 curl 验证 /generate 与
本地推理逐 token 一致、/generate/stream 逐 token 到达。

## 6. 汇报要求

1. 实际改动/新增文件清单
2. 每条验收命令输出摘要
3. 等价性证据与并发压测表（含 env 指纹、config 哈希、runs 目录）
4. 与规格的任何偏差及原因（无偏差写"无"）

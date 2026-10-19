# P3-T2 执行汇报（-o 文件）

规格：tasks/p3-t2-load.md；分支：codex/p3-report；提交前基线：87a19c0。

## 1. 实际新增文件清单

本次仅新增白名单路径，未修改已有文件、依赖或 src/prism，未删除文件、未 git push。

- scripts/serve_load.py：固定种子开放环到达、C槽客户端池、实际队长采样、T3判定、校准串行模拟适配与产物保存。
- configs/p3_load.json：18点网格、种子、连接池/超时、T3门槛、SHA-256固定的P1-T4校准输入。
- tests/test_serve_load.py：21项确定性、判定严格边界、与独立T3对照、有限C/超时、真实HTTP持久化、报告小节测试；不断言实测性能数字。
- results/p3/REPORT.md：PR #12部署等价性、T1流式与T2并发曲线、背压、雪崩偏差、客户端开销和边界。
- runs/20261019-p3-load-120854372978-f93d7f/：18点正式实验、校准原字节、源码快照、P0元数据、完整请求/队列、模拟对照、验收输出、本汇报与逐文件清单。
- runs/20261019-p3-load-121333469665-78a04d-check/：最终验收4点真实HTTP smoke，0预热/2重复；非实验证据。

全部新增路径逐项列于 [files.txt](files.txt)，共 244 个文件。
早期开发smoke及pytest临时产物未混入正式数据；早期仓库内smoke仍仅位于白名单runs下、保持gitignore，不提交。

## 2. 第5节每条验收命令输出摘要

| 命令 | 退出码 | 输出摘要 |
| --- | ---: | --- |
| uv sync --locked | 0 | Resolved 9 packages in 2ms；Checked 8 packages in 0.15ms |
| uv run --locked pytest | 0 | 666 passed in 20.35s（含本任务21项） |
| uv run --locked ruff check . | 0 | All checks passed! |
| uv run --locked ruff format --check . | 0 | 73 files already formatted |
| uv run --locked python scripts/serve_load.py --check | 0 | r40/r160 × c1/c8 × q2共4点，完整保存真实HTTP与模拟；smoke非实验证据 |
| git status --short | 0 | configs/p3_load.json、results/p3/、scripts/serve_load.py、tests/test_serve_load.py；runs受既有gitignore忽略，提交时仅定向强制添加上述正式/验收目录 |

完整逐条stdout/stderr与退出码：[acceptance/commands.json](acceptance/commands.json)，单条输出见acceptance/01.log–06.log。
补充核对：git diff --cached --check通过；18点P0配置哈希/样本数/计数守恒、同λ请求序列一致性、运行源码快照哈希全部通过。
最终独立验收仍由规划层执行。

## 3. 压测曲线与雪崩点（含env/config）

硬件/设备：Apple M4 Max / arm64 / CPU / 单原生线程；Python3.12.15、NumPy2.5.4。
λ={10,40,160}/s，C={1,8,64}，Q={2,64}；seed=20261019，窗口2000ms；每点预热1/重复3。
HTTP客户端池C槽、连接/读取超时10s，HTTP workers128/pending0；Connection: close，逐请求新建连接。
成功请求p50/p95/p99采用NumPy linear、包括dispatch与完整排空；P0整轮measurement另存，不当作请求分位数。
实测计数合计3轮，模拟为相同轨迹预热0/重复1，比较计数需除以3。

env 指纹：`bddd54542eebe6920420c857c2911fde714098898a8c2a68366c44dc6b6fade3`。
recipe config SHA-256：`900fd7982291e44d64cb386079e3cae1c4dfd3f25b6070a78ff7a2bd8c811fa6`。
逐点config哈希、全部18点曲线：[REPORT.md](REPORT.md)；机器可读：[summary.json](summary.json)。

7614次计划到达：成功2680、429=1579、客户端满=3355；超时/503/传输错误均0。
C=64、Q=64实测首触发λ=40/s（3/3轮同时触发p99和队长增长代理），校准模拟首触发λ=160/s；
离散首触发偏差+120/s（相对+300%），不是连续临界值。
40/s同点实测p50/p95/p99=714.850/985.048/1049.353ms，模拟p99=410.770ms，偏差−60.85%。
160/s同点实测p99=2173.525ms，模拟1654.919ms，偏差−23.86%；两侧都触发p99门槛。
40/s、Q=2、C=64实测p99=117.925ms且拒绝91/264；160/s同C/Q拒绝754/921，p99=159.010ms。
低并发的低尾时延伴随大量客户端满丢弃，不能解释为服务稳定。

雪崩判定原样复用T3 OR规则：p99>1000ms，或末半窗四段时间加权队长均值严格递增、斜率>1/s且终点队长≥20。
逐轮判定，不把3轮串接当增长轨迹。模拟与实测使用相同时刻/长度、λ/C/Q、seed/窗口/客户端超时；
串行适配器无限C时逐点与原T3完成/拒绝、分位数和判定一致；服务模型来自已归档P1-T4，未重拟合HTTP。
客户端CPU、dispatch、transport residual及队列采样间隔独立记录。实际最大采样间隔35.108ms，目标1ms，队长代理有采样误差。

边界：单进程串行推理、loopback、无 TLS、CPU小模型、本地线程竞争、无keepalive；T2只扫描整包，流式部署等价性引P3-T1 PR #12。
校准省略prefill截距、decode仅context32，模拟不含HTTP/429耗时；模拟结论和生产能力不混同。

## 4. 与规格的偏差及原因

无。

实现细节：T3本身不表达客户端并发池，因此仅在白名单脚本内加B=1 FIFO适配器，并在无限C条件下对独立T3做逐点一致性门控；
这用于落实同参数λ×C×Q对照，未改动src/prism或T3协议。Q采样是明确标注的实测近似，保留实际采样间隔供审查。

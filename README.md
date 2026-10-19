# Prism

棱镜：以可复现实验理解 AI 基础设施的设计、取舍与失效边界。

原始愿景见 [DESIGN](DESIGN.md)，阶段与验收见 [ROADMAP](docs/ROADMAP.md)，
双 agent（Kimi 规划把关 + Codex 执行）协作约定见 [AGENTS](AGENTS.md)。

当前处于立项骨架阶段，P0（测量地基）尚未开始。

## 运行

环境：Python 3.12，uv，Apple Silicon Mac（CPU/MPS）；无 GPU/CUDA 依赖。

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m prism
```

## 目录

- `src/prism/` — 实现代码
- `tests/` — 测试
- `configs/` — 实验配方配置
- `results/` — 阶段验收报告与正式证据摘要
- `runs/` — 单次运行产物（不入库）
- `docs/` — ROADMAP 与设计文档

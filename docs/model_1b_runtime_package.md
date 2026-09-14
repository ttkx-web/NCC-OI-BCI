# 1B Runtime Model Package

一个 frozen 1B flatten linear head 对应一个固定窗口长度的 Runtime Model Package。
预训练 backbone 始终保留 10 个 time-position embeddings，但 Package 的实际窗口可以是
1–10 个整秒 patch；`window_sec`、`num_time_patches`、`token_count` 与
`classifier_input_dim` 全部从 head checkpoint metadata 导出，不能由 Runtime 改写。

当前 4 秒群体 head 的导出示例：

```bash
python scripts/export_1b_model_package.py \
  --backbone-checkpoint checkpoints/backbones/1b/pretrain_checkpoint_4.pt \
  --head-checkpoint /path/to/1b_4s_population_head.pt \
  --output-dir model_packages/1b/bnci2014_001/subject_01/population/4s_flatten/v1 \
  --device cuda
```

该 Package 会记录 `window_sec=4`、`num_time_patches=4`、`token_count=256` 和
`classifier_input_dim=524288`。以后训练出 1/2/3/10 秒 head 时，只需替换
`--head-checkpoint` 和对应输出目录；无需修改 exporter、loader 或 Runtime。

## Offline replay and latency acceptance

The formal 4-second Package can use the existing offline replay path directly;
it does not use Neuracle acquisition, HTTP, or network transport:

```bash
python scripts/replay_offline.py \
  --config configs/stage1/replay_1b_4s.yaml \
  --device cuda
```

Runtime 会拒绝不匹配 Package `window_sec` 的输入，且不会补零、裁剪或以另一长度 head
代替。用 `--max-windows 3 --no-jsonl-log --summary-json runs/.../smoke_summary.json`
可做小规模 replay smoke，避免覆盖默认 replay 输出。

窗口延迟 benchmark 也复用通用 Package loader、`ReplayWindowProvider` 和
`RuntimeBenchmarkCore`。HDF5 在 provider 迭代开始时预加载；HDF5 读取不计入每窗口
`compute_total_ms`。先在服务器上执行 2 warmup + 3 measured windows 的 smoke：

```bash
python scripts/run_window_latency_benchmark.py \
  --config configs/benchmarks/window_latency_1b_4s.yaml \
  --device cuda \
  --warmup-windows 2 \
  --measured-windows 3 \
  --output-root runs/benchmarks/smoke \
  --run-id model_1b_4s_smoke
```

正式 20/200 验收：

```bash
python scripts/run_window_latency_benchmark.py \
  --config configs/benchmarks/window_latency_1b_4s.yaml \
  --device cuda \
  --output-root runs/benchmarks/acceptance \
  --run-id model_1b_4s_20_200
```

同机、同一 HDF5/session/window/step 的 50M vs 1B 成对比较：

```bash
python scripts/run_window_latency_benchmark.py \
  --config configs/benchmarks/window_latency_50m_1b_4s.yaml \
  --device cuda \
  --output-root runs/benchmarks/acceptance \
  --run-id model_50m_1b_4s_20_200
```

每次换用新的 `--run-id` 或 `--output-root`，因为 benchmark 会拒绝覆盖已有 run 目录。
读取 `summary.json` 中每个 candidate 的 `preprocessing_ms`、`inference_ms`、
`output_materialization_ms` 和 `compute_total_ms` 的 `p50`、`p95`、`max`；对 1B，
`inference_ms` 包含最终 encoder layer 与正式 linear head，不是此前的 backbone-only
latency-only 指标。

本阶段仅验收离线 HDF5 replay 的预测路径和计算延迟；不代表 Neuracle 采集、网络传输、
HTTP 或真实设备端到端延迟。下一阶段才涉及更高级的部署服务与个人化策略。

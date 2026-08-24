# 三状态 localhost inference service

`scripts/serve_inference.py` exposes the formal three-head Runtime Model Package
as a local IPC service. It listens on `127.0.0.1:8767` by default and loads the
package once during startup; every request reuses that predictor instance.

## Start

```bash
python scripts/serve_inference.py \
  --model-package model_packages/50m_three_mental_states \
  --host 127.0.0.1 \
  --port 8767 \
  --device cpu
```

`GET /health` returns `status`, `model_loaded`, package path, and device.

`POST /infer` accepts contract v1 JSON. `eeg` is strictly `[C,T]`, its first
dimension must match `channel_names`, values must be finite, the unit must be
`uV`, and `T == sequence_end - sequence_start + 1`.

```json
{
  "schema_version": "1.0",
  "sample_rate_hz": 250,
  "unit": "uV",
  "channel_names": ["C3", "C4"],
  "sequence_start": 10000,
  "sequence_end": 10002,
  "eeg": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
}
```

The response preserves the sequence bounds and contains named model tasks. Each
prediction includes `task_id`, `class_id`, `label`, `confidence`, and the
predictor's original `probabilities`. `latency_ms` measures only the service's
core inference call: it excludes HTTP transport, request parsing, and JSON
serialization.

The client owns device decoding, conversion to `uV`, and extracting one complete
window. NCC-OI-BCI owns channel adaptation, resampling, filtering,
normalization, model preprocessing, and model inference. The service does not
perform sliding-window segmentation.

## 三状态调用链与脚本

任务顺序固定为 `workload`、`attention`、`emotion`。各任务的类别名称与输出维度来自 Package metadata；单窗口始终只进行一次 Python 预处理、一次共享 50M Backbone forward 和每个 Head 一次 forward。

```text
HDF5 / client [C,T] window → Python channel adaptation / resample / filter / normalize
→ shared 50M Backbone → workload, attention, emotion heads → JSON response
```

导出自包含 Package：

```bash
python scripts/export_50m_multi_head_model_package.py --output-dir model_packages/50m_three_mental_states
```

正式三状态脚本只有四个：

- `scripts/export_50m_multi_head_model_package.py`
- `scripts/serve_inference.py`
- `scripts/run_multi_head_trials.py`
- `scripts/verify_three_state_inference.py`

统一验证脚本支持 `--mode direct|package|decoder|http|all`，并比较完整 probability vector。`http` 默认启临时 localhost 服务，也可用 `--server-url` 指向已启动服务；`--export-request` 和 `--export-reference` 可导出 fixture。

```bash
python scripts/verify_three_state_inference.py \
  --mode http \
  --model-package model_packages/50m_three_mental_states \
  --input-h5 data/processed/bnci2014_001/subject_01.h5
```

此前分散的 direct、Package、Decoder 和 HTTP smoke 验证已分别合并到上述 `direct`、`package`、`decoder` 和 `http` mode。

Rust/设备端只负责采集、窗口切分、单位转换和 HTTP 调用；Python Runtime 仍负责通道适配、重采样、滤波、归一化和模型推理。服务不负责滑窗切分。

To make a real-data direct-vs-HTTP check with the same exact window:

```bash
python scripts/verify_three_state_inference.py --mode http \
  --model-package model_packages/50m_three_mental_states \
  --input-h5 data/processed/bnci2014_001/subject_01.h5 \
  --device cpu
```

It starts an ephemeral localhost server unless `--server-url` is supplied, then
compares task IDs, class IDs, labels, confidences, and complete probability
vectors at `rtol=1e-5, atol=1e-6`.

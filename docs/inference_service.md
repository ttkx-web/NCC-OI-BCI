# Localhost inference service

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

To make a real-data direct-vs-HTTP check with the same exact window:

```bash
python scripts/test_inference_service_offline.py \
  --model-package model_packages/50m_three_mental_states \
  --input-h5 data/processed/bnci2014_001/subject_01.h5 \
  --device cpu
```

It starts an ephemeral localhost server unless `--server-url` is supplied, then
compares task IDs, class IDs, labels, confidences, and complete probability
vectors at `rtol=1e-5, atol=1e-6`.

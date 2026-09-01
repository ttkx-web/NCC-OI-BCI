# NCC-OI-BCI

面向非侵入式 EEG BCI 的 **统一模型 Runtime、离线/伪实时评估、个体化适配与真实设备实时推理框架**。

当前主线任务为 **BNCI2014_001 四分类运动想象**，核心模型为公司 **50M EEG 基座模型**；LaBraM 作为已接入的公开基线。项目已经完成从离线数据、Runtime Model Package、Replay/Streamlit，到 **Neuracle JellyFish 实时采集 → 4 秒滑窗 → 50M 推理** 的统一链路。

> 当前状态：Unified Runtime 已完成；50M 与 LaBraM 已接入统一 Package/Runtime；Stage 1A 个体化分类头已实现；Stage 2A 自采 Neuracle 离线数据处理已实现；Stage 2B 真实设备实时采集、Trigger、连续滑窗与 50M 实时推理闭环已实现。Rest-Tune 与 NeuroOnline 目前仅保留接口/基础类型，真实算法仍待开发。

---

## 1. 当前任务

BNCI2014_001 四分类运动想象：

| EEG 类别 | 控制命令 |
|---|---|
| `left_hand` | `LEFT` |
| `right_hand` | `RIGHT` |
| `feet` | `FORWARD` |
| `tongue` | `STOP` |

当前主要模型：

- **50M**：公司 EEG 基座模型，当前实时部署主模型；
- **LaBraM**：已完成统一 Runtime 接入的公开基线；
- **CBraMod**：计划作为下一阶段基线接入；
- BIOT / REVE：按后续实验需要再扩展。

---

## 2. 阶段进度

| 阶段 | 目标 | 状态 |
|---|---|---|
| Stage 0 | LaBraM 离线/伪实时基础 Pipeline | 已完成 |
| Stage 0.5 | 50M 替换、正式分类头、Replay、Streamlit | 已完成 |
| Unified Runtime | 统一 Raw EEG → Canonical → Transform → Backend → ModelOutput | 已完成 |
| Stage 1A | 冻结 50M Backbone 的群体头与个人头 | 已完成 |
| Stage 1B | 解冻部分 Backbone 的任务相关个体化 | 计划中 |
| Stage 2A | Neuracle 自采离线 EEG / BDF / Marker / 4s HDF5 导出 | 已实现 |
| Stage 2B | Neuracle JellyFish 真实实时接入、Trigger、4s Window、50M 推理 | 已实现 |
| Stage 3 | 实时 Pipeline 与个体化策略统一 | 计划中 |
| Rest-Tune | 离线个体化 Backbone/Head 适配 | 接口已预留，算法待实现 |
| NeuroOnline | 基于在线反馈的持续适配 | 接口已预留，算法待实现 |

---

## 3. 总体架构

### 3.1 离线 / Replay

```text
HDF5 / Recorded EEG
        |
        v
RawEEGWindow
        |
        v
SignalCanonicalizer
        |
        v
ModelInputTransform
├── Model50MInputTransform
└── LaBraMInputTransform
        |
        v
PreparedModelInput
        |
        v
ModelBackend
├── Model50MBackend
└── LaBraMBackend
        |
        v
RuntimeModel
        |
        v
ModelOutput
        |
        +--> RuntimeEvaluator
        +--> CLI Replay
        +--> Streamlit
        +--> JSONL / Summary / Latency
```

### 3.2 真实实时链路

```text
Neuracle EEG Device
        |
        v
Neuracle acquisition / JellyFish
        |
        v  TCP
NeuracleJellyFishSource
        |
        v
65ch mixed stream
59 EEG + 4 EOG + 1 ECG + 1 Trigger
        |
        v
select_verified_eeg_channels()
        |
        v
59ch EEG / 1000 Hz / uV
        |
        v
RealtimeEEGWindowPipeline
        |
        +--> TimestampedRingBuffer
        +--> gap / continuity check
        +--> Trigger association
        |
        v
RealtimeWindow [59, 4000]
4.0 s window / 0.5 s step
        |
        v
RealtimeRuntimeBridge
        |
        v
RawEEGWindow
        |
        v
RuntimeModel.prepare()
        |
        v
50M PreparedModelInput
signal [1, 64, 400]
channel_valid_mask [1, 64]
        |
        v
RuntimeModel.predict_prepared()
        |
        v
50M Backbone + Linear Head
        |
        v
predicted_class / probabilities / confidence
        |
        +--> runtime_predictions.jsonl
        +--> runtime_inference_summary.json
```

实时链路不重新实现模型预处理。`RealtimeRuntimeBridge` 只负责 Source/Prepared Input 安全 Gate，真正的通道映射、重采样、滤波、归一化与模型输入构造仍由统一 `RuntimeModel.prepare()` 负责。

---

## 4. Unified Runtime

核心接口：

```text
src/bci_dayloop/runtime/
├── types.py
├── model.py
└── adaptation_types.py

src/bci_dayloop/preprocessing/
├── canonical.py
├── model_50m.py
└── labram.py

src/bci_dayloop/models/
├── base.py
├── labram_backend.py
└── model_50m/backend.py
```

统一调用方式：

```python
prepared = runtime_model.prepare(raw_window)
output = runtime_model.predict_prepared(prepared)
```

上层 Replay、Evaluator 和 UI 不需要知道当前模型是 50M 还是 LaBraM。

---

## 5. Runtime Model Package（schema v2）

当前统一 Runtime Package 为自包含五文件结构：

```text
<package>/
├── package.yaml
├── backbone.pt
├── classifier.pt
├── preprocessing.yaml
└── metrics.json
```

`package.yaml` 包含：

- `schema_version: 2`；
- 模型类型与任务信息；
- `InputContract`；
- Runtime `step_sec` / confidence threshold；
- command map；
- Backbone / Classifier 文件哈希；
- offline / online adaptation metadata。

当前 Loader 支持：

```text
model.type = model_50m
model.type = labram
```

统一入口：

```python
from bci_dayloop.packages import load_runtime_package

package = load_runtime_package(
    "model_packages/...",
    device="cpu",
)

runtime_model = package.runtime_model
```

旧版 `model.yaml/base_model.json` Package 仅作为历史兼容代码保留，新开发应使用 schema-v2 Package。

---

## 6. 当前 50M 4 秒 Runtime 契约

BNCI2014_001 / Stage 1 与 Stage 2B 当前使用的 50M 下游契约：

| 配置项 | 当前值 |
|---|---|
| 任务 | 四分类运动想象 |
| 模型目标采样率 | 100 Hz |
| 标准目标通道 | 64 |
| Runtime window | 4.0 s |
| Runtime step | 0.5 s |
| 模型输入 | `[1, 64, 400]` |
| Patch 长度 | 1 s |
| Patch 步长 | 1 s |
| 实际时间 Patch 数 | 4 |
| 预训练时间位置容量 | `model_n_time_patches=10` |
| Transformer 输出层 | `output_layer_idx=8` |
| 默认聚合 | `flatten` |
| 分类类别数 | 4 |

### 6.1 为什么 4 秒输入仍保留 10 个时间位置

50M Backbone 的预训练 checkpoint 使用 10 秒时间位置参数。4 秒下游输入只产生 4 个真实时间 Patch：

```text
实际输入时间 Patch = 4
预训练时间位置容量 = 10
实际 Token 数 = 64 × 4 = 256
```

`model_n_time_patches=10` 不代表将 4 秒 EEG 补成 10 秒。

---

## 7. Stage 2B 实时输入契约

当前批准的实时设备输入为：

```text
59 EEG channels
1000 Hz
uV
4.0 s
[59, 4000]
```

Runtime 目标：

```text
64 channels
100 Hz
4.0 s
[1, 64, 400]
```

当前批准的 59 → 64 mapping：

- 同名有效目标通道：57；
- 设备额外通道：`PO5`, `PO6`，允许忽略；
- 50M 目标缺失通道：`AFz`, `CPz`, `P1`, `P2`, `Iz`, `F9`, `F10`；
- 缺失目标由统一 50M preprocessing 显式 zero-fill；
- 对应 `channel_valid_mask=False`；
- 其他 alias、重复通道、未知通道或额外缺失通道均 fail closed。

Source 层 `model_safe=true` 只代表实时 EEG 的单位/通道来源通过验证，不代表最终模型输入已通过；最终仍必须经过 `RealtimeRuntimeBridge` 的 prepared-input gate。

> 当前 Stage 2B 实时 Gate 固定验证 4 秒 50M Runtime Contract。后续 2s / 3s / 4s 延迟实验需要同步扩展 Runtime Package 与 realtime gate，不能直接拿 4s Package 跑 2s 输入。

---

## 8. BNCI2014_001 数据与实验协议

处理后的数据建议按被试保存：

```text
data/processed/bnci2014_001/
├── subject_01.h5
├── subject_02.h5
├── ...
└── subject_09.h5
```

### 8.1 群体模型 LOSO baseline

以 Subject 1 为目标被试：

```text
Population Train:
Subject 2–9 / 0train

Population Validation:
Subject 2–9 / 1test

Final Held-out Test:
Subject 1 / 1test
```

目标被试不得参与群体模型训练和模型选择。

### 8.2 Personal Head

```text
Target Subject / 0train
        |
        +--> Personal Train
        +--> Personal Validation

Target Subject / 1test
        |
        +--> Final Test
```

Stage 1A 当前只更新分类头，50M Backbone 保持冻结。

### 8.3 正式性能评估

正式 Accuracy / Balanced Accuracy / Macro-F1 / Confusion Matrix 使用：

```text
scripts/evaluate_runtime_package.py
```

Evaluator 为 trial-aligned：每个 trial 独立切窗，不允许窗口跨 trial。

连续 Replay 主要用于工程链路与 latency 验证，不应替代正式 trial-aligned 性能评价。

---

## 9. 目录结构

```text
NCC-OI-BCI/
├── configs/
│   ├── stage0/
│   ├── stage05/
│   └── stage1/
│
├── data/
│   ├── raw/
│   ├── cache/
│   ├── moabb_cache/
│   └── processed/
│       └── bnci2014_001/
│
├── checkpoints/
│   ├── backbones/
│   │   ├── 50m/
│   │   └── labram/
│   └── heads/
│
├── model_packages/
├── registries/
├── runs/
│   ├── stage1/
│   ├── stage2a/
│   └── stage2b/
│
├── docs/
│   ├── stage2a/
│   └── stage2b/
│
├── scripts/
├── src/bci_dayloop/
│   ├── acquisition/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── packages/
│   ├── personalization/
│   ├── preprocessing/
│   ├── realtime/
│   ├── runtime/
│   └── vendor/
│
├── tests/
├── web/
└── README.md
```

目录职责：

| 目录 | 内容 |
|---|---|
| `data/` | 原始、缓存与处理后的 EEG 数据 |
| `checkpoints/backbones/` | 基座模型 checkpoint |
| `checkpoints/heads/` | 训练后的任务分类头 |
| `model_packages/` | schema-v2 Runtime Model Package |
| `runs/` | 实验日志、指标、prediction 与 summary |
| `registries/` | Stage 1 个人模型 Registry |
| `docs/stage2a/` | 自采离线数据 SOP / 审计 |
| `docs/stage2b/` | 实时设备、Trigger、Window 与 Runtime Contract 验证记录 |

---

## 10. 环境安装

项目使用 Python 3.11。

### 10.1 基础环境

```bash
git clone https://github.com/ttkx-web/NCC-OI-BCI.git
cd NCC-OI-BCI

conda create -n bci-dayloop python=3.11 -y
conda activate bci-dayloop
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 10.2 NVIDIA CUDA 环境

```bash
conda env create -f environment.yml
conda activate bci-dayloop
python -m pip install -e .
```

### 10.3 BNCI 数据准备环境

MOABB 只用于数据下载/准备，建议独立环境：

```bash
python -m pip install -r requirements-data.txt
```

### 10.4 Neuracle BDF / JellyFish 依赖

当前 vendor 代码会使用 `pyedflib`。若本地未安装：

```bash
python -m pip install pyedflib
```

后续应将该依赖正式加入项目依赖或改为延迟导入，避免新环境运行 Stage 2A/2B 时出现缺包。

---

## 11. 准备 BNCI2014_001

生成处理后的 HDF5：

```bash
python scripts/prepare_bnci2014_001.py --help
```

检查：

```bash
python scripts/inspect_dataset.py \
  data/processed/bnci2014_001/subject_01.h5
```

大文件、模型权重和运行产物默认不提交 Git。

---

## 12. 训练 50M 群体头

```bash
python scripts/train_50m_population_head.py \
  --data-root data/processed/bnci2014_001 \
  --data-pattern "subject_{subject:02d}.h5" \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subject 1 \
  --train-session 0train \
  --validation-session 1test \
  --final-test-session 1test \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --device cpu \
  --window-sec 4 \
  --window-stride-sec 4 \
  --window-construction direct_trial \
  --model-n-time-patches 10 \
  --target-sample-rate 100 \
  --patch-sec 1 \
  --patch-stride-sec 1 \
  --output-layer-idx 8 \
  --aggregation flatten \
  --feature-batch-size 1 \
  --epochs 100 \
  --head-batch-size 32 \
  --head-lr 0.001 \
  --weight-decay 0.001 \
  --metric-for-best val_bacc \
  --patience 15
```

正式实验不要使用：

```text
--max-windows-per-class-per-subject
```

`--save-feature-cache` 仅在需要将 frozen feature 写盘时开启；默认可只在内存中训练分类头。

---

## 13. 训练 LaBraM 群体头

```bash
python scripts/train_labram_population_head.py \
  --data-root data/processed/bnci2014_001 \
  --data-pattern "subject_{subject:02d}.h5" \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subject 1 \
  --train-session 0train \
  --validation-session 1test \
  --final-test-session 1test \
  --checkpoint checkpoints/backbones/labram/labram_base.pth \
  --device cpu \
  --embedding-batch-size 1 \
  --window-sec 4 \
  --target-sample-rate 200 \
  --patch-samples 200 \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --patience 12 \
  --overwrite
```

LaBraM Backbone 冻结，先提取 frozen embeddings，再训练 Linear Head。

---

## 14. 导出 Runtime Model Package

### 14.1 50M

```bash
VERSION=$(date +%Y%m%d_%H%M%S)

python scripts/export_50m_model_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --classifier checkpoints/heads/stage1/bnci2014_001/subject_01/population/4s_flatten/head.pt \
  --output model_packages/stage1/bnci2014_001/subject_01/population/4s_flatten/${VERSION} \
  --device cpu \
  --session 1test \
  --step-sec 0.5
```

### 14.2 LaBraM

```bash
VERSION=$(date +%Y%m%d_%H%M%S)

python scripts/export_labram_model_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --checkpoint checkpoints/backbones/labram/labram_base.pth \
  --classifier <labram-population-head.pt> \
  --output model_packages/stage0/bnci2014_001/subject_01/population/4s_labram/${VERSION} \
  --device cpu \
  --session 1test \
  --step-sec 0.5
```

导出脚本会构建 schema-v2 Package，并默认执行 Package reload / smoke test。

---

## 15. Trial-aligned 正式评估

### 15.1 50M

```bash
python scripts/evaluate_runtime_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package <50m-package> \
  --session 1test \
  --device cpu \
  --step-sec 4 \
  --output runs/evaluation/50m_population_s01.json
```

### 15.2 LaBraM

```bash
python scripts/evaluate_runtime_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package <labram-package> \
  --session 1test \
  --device cpu \
  --step-sec 4 \
  --output runs/evaluation/labram_population_s01.json
```

输出包括：

```text
Window-level Accuracy / BAcc / Macro-F1
Trial-level Accuracy / BAcc / Macro-F1
Confusion Matrix
Preprocessing latency
Model latency
Total P50 / P95 latency
```

对于当前 4 秒 direct-trial 正式比较，建议优先报告 `trial_metrics`。

---

## 16. CLI Replay

```bash
python scripts/replay_offline.py \
  --config configs/stage1/replay_population_4s.yaml \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package <runtime-package> \
  --device cpu \
  --max-windows 20 \
  --replay-speed 100
```

CLI Replay 用于验证：

- Runtime Package 加载；
- 连续滑窗；
- prediction / confidence；
- preprocessing / model / total latency；
- JSONL 与 summary。

正式分类性能优先使用 `evaluate_runtime_package.py`。

---

## 17. Streamlit

```bash
streamlit run web/app.py
```

当前页面支持：

- Runtime Package 选择；
- HDF5 Replay；
- Start / Stop / Restart；
- EEG waveform；
- 当前 prediction / confidence / command；
- Prediction History；
- current / average / P95 latency；
- window completion / failure；
- 可选 JSONL logging。

当前 Streamlit 主要用于 HDF5 Replay；Stage 2B 真实设备推理使用独立实时 Probe。

---

## 18. Stage 2A：Neuracle 离线数据

仓库已提供：

```text
scripts/inspect_neuracle_bdf.py
scripts/inspect_neuracle_trials.py
scripts/align_neuracle_collect.py
scripts/audit_neuracle_dataset.py
scripts/export_neuracle_stage2a_4s.py
```

当前验证过的 Neuracle 导出中，`1.bdf` 同时包含连续信号与完整事件，可作为标准离线输入。具体流程见：

```text
docs/stage2a/neuracle_export_sop.md
```

---

## 19. Stage 2B：真实 Neuracle → 50M 推理

### 19.1 Source Probe

```bash
python scripts/probe_neuracle_realtime.py --help
```

用于验证：

```text
JellyFish connection
META
packet continuity
EEG / EOG / ECG / Trigger metadata
reconnect / stop
```

### 19.2 Window Probe

```bash
python scripts/probe_neuracle_window_pipeline.py --help
```

用于验证：

```text
EEG-only selection
Timestamped buffer
4s / 0.5s windowing
Trigger-window association
gap / overflow / completion
```

### 19.3 真实实时 50M 推理

```bash
python scripts/probe_neuracle_runtime_inference.py \
  --package <50m-runtime-package> \
  --device cpu \
  --duration-sec 60 \
  --host 127.0.0.1 \
  --port 8712 \
  --expected-sfreq 1000 \
  --window-sec 4 \
  --step-sec 0.5
```

当前该 Probe **只接受 `model_50m` Runtime Package**，并拒绝 test head。

成功窗口执行：

```text
RealtimeWindow
→ RealtimeRuntimeBridge.prepare
→ model_input_safe gate
→ RuntimeModel.predict_prepared
→ 50M Backbone
→ Linear Head
→ prediction
```

默认输出：

```text
runs/stage2b/neuracle_runtime_inference/
├── runtime_predictions.jsonl
└── runtime_inference_summary.json
```

逐窗口记录：

```text
window_id
continuous_segment_id
source_shape
prepared_shape
valid_channel_count
predicted_class
predicted_name
confidence
probabilities
prepare_latency_ms
inference_latency_ms
total_model_latency_ms
marker_summary
```

Summary 重点检查：

```text
status == passed
missing_packets == 0
duplicate_packets == 0
out_of_order_packets == 0
gap_count == 0
model_input_failure_count == 0
prediction_failure_count == 0
prediction_success_count == emitted_windows
```

Stage 2B Probe 不持久化 EEG waveform。

---

## 20. Personalization / Adaptation

### 20.1 已实现

- Population Head；
- Personal Head；
- Personal Model Package；
- Personal Model Registry；
- Population vs Personal Replay；
- `OfflineAdaptationStrategy` 基础接口；
- `OnlineAdaptationStrategy` 基础接口；
- `NoOfflineAdaptation` / `NoOnlineAdaptation`。

### 20.2 尚未实现

- Rest-Tune 真实训练策略；
- NeuroOnline buffer / feedback update；
- 在线 optimizer / rollback；
- Rest-Tune + NeuroOnline 联合策略。

当前推荐开发顺序：

```text
50M / LaBraM / CBraMod baseline
        ↓
2s / 3s / 4s 性能与 latency benchmark
        ↓
选择 Demo 主模型与窗口
        ↓
NeuroOnline v0
frozen backbone + supervised online head update
        ↓
Rest-Tune
        ↓
Rest-Tune + NeuroOnline
```

---

## 21. 测试

### 21.1 编译

```bash
python -m compileall -q src scripts web tests
```

### 21.2 Unified Runtime / 模型回归

```bash
python -m pytest -q \
  tests/test_model_50m_4s_runtime.py \
  tests/test_runtime_package.py \
  tests/test_replay_offline_cli.py \
  tests/test_runtime_control.py \
  tests/test_ui_runtime.py
```

### 21.3 Stage 2B 专项

```bash
python -m pytest -q \
  tests/test_neuracle_jellyfish.py \
  tests/test_realtime_contracts.py \
  tests/test_realtime_source_buffer.py \
  tests/test_realtime_windowing.py \
  tests/test_realtime_eeg_window_pipeline.py \
  tests/test_realtime_runtime_mapping.py \
  tests/test_realtime_runtime_bridge.py \
  tests/test_probe_neuracle_runtime_inference.py
```

### 21.4 全量

```bash
python -m pytest -q
```

合并主分支前要求：

```text
0 failed
0 errors
```

---

## 22. 合并 / Release 验收

```text
[ ] python -m compileall -q src scripts web tests
[ ] python -m pytest -q
[ ] 50M Runtime Package 可导出并重新加载
[ ] LaBraM Runtime Package 可导出并重新加载
[ ] trial-aligned evaluator 可输出 Accuracy/BAcc/Macro-F1/Confusion Matrix
[ ] CLI Replay 正常
[ ] Streamlit Start / Stop / Restart 正常
[ ] Stage 2B source/window/runtime inference 专项测试通过
[ ] 真设备 Probe 无 packet gap / duplicate / out-of-order
[ ] model_input_failure_count == 0
[ ] prediction_failure_count == 0
[ ] prediction_success_count == emitted_windows
[ ] Git 不包含 EEG 数据、Checkpoint、Runtime Package、运行结果或隐私信息
```

---

## 23. 已知限制

- Stage 2B 当前实时推理入口只允许 **50M**；LaBraM 尚未接入真实 JellyFish Probe；
- Stage 2B 当前批准的 Runtime Gate 固定为 **4s / 1000Hz source → 4s / 100Hz 50M**；2s/3s 实时 Contract 尚需扩展；
- Stage 1A 个体化目前只训练任务 Head；
- Rest-Tune 与 NeuroOnline 只有接口，尚未形成算法闭环；
- Streamlit 当前仍以 HDF5 Replay 为主，不直接消费 JellyFish 实时 Source；
- 实时 Trigger 当前进入窗口 metadata / prediction record，但任务级 Trigger code → class label 映射应由具体实验协议定义，不能在设备层硬编码；
- macOS 上 LaBraM/宽 Linear Head 优先使用 CPU，当前环境下 MPS 可能不稳定；
- `pyedflib` 尚未写入主依赖文件，新环境使用 Neuracle vendor 代码前需单独安装。

---

## 24. 重要文档

```text
docs/README_stage0.md
docs/README_stage05.md
docs/STAGE1_PERSONALIZATION.md

docs/stage2a/neuracle_export_sop.md

docs/stage2b/device_realtime_interface_audit.md
docs/stage2b/neuracle_material_audit.md
docs/stage2b/neuracle_real_device_validation.md
docs/stage2b/neuracle_realtime_unit_evidence.md
docs/stage2b/neuracle_trigger_validation.md
docs/stage2b/neuracle_window_pipeline_validation.md
docs/stage2b/realtime_model_input_contract_audit.md
docs/stage2b/realtime_runtime_input_contract.md
```

部分 Stage 2B audit 文档记录的是开发中间阶段的 fail-closed 状态，例如“仅允许 prepare、禁止 inference”。当前最新实时推理入口以：

```text
scripts/probe_neuracle_runtime_inference.py
```

以及对应测试：

```text
tests/test_probe_neuracle_runtime_inference.py
```

为准。

---

## 25. Git 与隐私边界

以下内容不得提交：

```text
data/raw/**
data/cache/**
data/moabb_cache/**
data/processed/**
checkpoints/**
runs/**
model_packages/**
registries/**
```

真实设备验证文档与日志不得包含：

- EEG waveform / 原始样本值；
- 受试者身份信息；
- 设备序列号；
- 私有网络地址；
- 本机绝对路径；
- 未脱敏设备标识。

---

## 26. 当前开发重点

当前框架层已经进入相对稳定阶段。后续模型端优先：

1. 固定 50M / LaBraM baseline；
2. 接入 CBraMod；
3. 比较 2s / 3s / 4s 窗口下的 BAcc、Macro-F1、prepare latency、inference P50/P95；
4. 选择单被试 Demo 主模型与窗口长度；
5. 实现 NeuroOnline v0；
6. 再推进 Rest-Tune 与 Rest-Tune + NeuroOnline。

设备端继续负责 JellyFish、数据连续性、时间戳与 Trigger；模型端从 `RealtimeWindow / RawEEGWindow` 之后负责窗口配置、Runtime、模型、适配与评估。

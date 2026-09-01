# NCC-OI-BCI

基于公司 **50M EEG 基座模型** 的离线、伪实时与实时脑机接口基础设施。

本仓库提供统一的 EEG 数据读取、预处理、模型适配、Runtime Model Package、滑窗推理、运行控制、日志记录和 Streamlit 可视化能力，用于逐步完成从公开离线数据验证到公司自采数据、真实设备接入、个体化和在线自适应的工程闭环。

> 当前状态：**Stage 0.5 已完成**。当前并行推进 **Stage 1：简单个体化** 与 **Stage 2A：公司自采离线 EEG 数据兼容**。

---

## 1. 项目目标

NCC-OI-BCI 的目标不是只运行某一个固定模型，而是构建一套可复用的 BCI Runtime，使不同 EEG 数据源、模型和任务能够通过统一接口接入同一条推理链路。

当前主线任务为 **BNCI2014_001 四分类运动想象**：

| 类别 | 控制命令 |
|---|---|
| `left_hand` | `LEFT` |
| `right_hand` | `RIGHT` |
| `feet` | `FORWARD` |
| `tongue` | `STOP` |

当前正式运行模型为公司 **50M EEG 基座模型**。Stage 0 使用的 LaBraM 作为历史基线保留，用于回归测试和模型替换前后的工程对照。

---

## 2. 当前进度

| 阶段 | 目标 | 状态 |
|---|---|---|
| Stage 0 | LaBraM 伪实时基础 Pipeline | 已完成 |
| Stage 0.5 | 将 LaBraM 替换为 50M，并完成正式分类头、Model Package、CLI Replay 和 Streamlit 验证 | 已完成 |
| Stage 1 | 50M 简单个体化及伪实时验证 | 当前推进 |
| Stage 2A | 公司自采离线 EEG 数据兼容 | 当前推进 |
| Stage 2B | EEG 设备真实实时接入 | 计划中 |
| Stage 3 | 实时 Pipeline 与简单个体化合并 | 计划中 |
| Stage 3.5 | Rest-Tuning 静息态个体化 | 计划中 |
| Stage 4 | 实时自适应 | 计划中 |

---

## 3. 已实现能力

### 3.1 模型与预处理

- 公司 50M EEG Backbone 构建、冻结和部署 checkpoint 加载；
- 标准 64 通道映射与通道别名归一化；
- 缺失通道补零及 `channel_valid_mask`；
- EEG 单位统一；
- 0.1–75 Hz 带通滤波；
- 100 Hz 重采样；
- 严格 10 秒窗口检查；
- 有效通道按时间维 Z-score；
- `[64, 1000] -> [640, 100]` Token 化；
- 指定 Transformer Block 的 Token Embedding 提取；
- Flatten / Mean 特征聚合；
- 线性任务分类头训练、保存和加载；
- `predict_proba()`、`predict()` 和 `extract_embeddings()`；
- 字典式 `signal + channel_valid_mask` 通用模型输入。

### 3.2 Runtime 与 Pipeline

- 模型无关的 Model Adapter / Model Factory；
- Runtime Model Package 保存和重新加载；
- HDF5 离线数据伪实时回放；
- 固定窗口、固定步长滑窗推理；
- CLI Replay；
- Streamlit 可视化；
- Start / Stop / Restart 运行控制；
- 预测类别、概率、置信度和命令映射；
- 窗口计数与异常状态记录；
- 当前、平均和 P95 延迟记录；
- JSONL 窗口日志；
- Summary JSON 运行报告。

### 3.3 测试与验证

- 50M Tokenization Smoke Test；
- 50M Backbone Smoke Test；
- 50M Classifier Smoke Test；
- 50M Adapter Smoke Test；
- 50M Runtime Package Smoke Test；
- BNCI 真实 10 秒离线窗口测试；
- CLI Replay 验证；
- Streamlit Start / Stop / Restart 验证；
- LaBraM 历史 Pipeline 回归测试；
- 单元测试与集成测试。

---

## 4. 系统架构

```text
EEG Data Source
├── Public offline EEG
├── Self-collected offline EEG
└── Realtime EEG stream
        |
        v
Acquirer / Replay
        |
        v
Window Buffer and Sliding Window
        |
        v
Model-specific Pipeline Preprocessor
├── 50M Preprocessor
└── LaBraM Preprocessor
        |
        v
Model Adapter
├── 50M Adapter        <- current default
└── LaBraM Adapter     <- Stage 0 legacy baseline
        |
        v
Runtime Model Package
        |
        v
SlidingWindowDecoder
        |
        v
PipelineController
├── CLI Replay
├── Streamlit UI
├── JSONL Window Log
└── Summary / Latency Metrics
```

Pipeline 与具体模型解耦。模型所需的采样率、窗口长度、通道顺序、预处理方式和分类头参数由对应的 Model Adapter 与 Runtime Model Package 提供，不应硬编码在通用滑窗和展示模块中。

---

## 5. 当前 50M Runtime 契约

### 5.1 当前配置

| 配置项 | 当前值 |
|---|---|
| 数据集 | BNCI2014_001 Subject 1 |
| 任务 | 四分类运动想象 |
| 原始采样率 | 250 Hz |
| Runtime 窗口 | 10 秒 |
| 滑窗步长 | 0.5 秒 |
| 目标采样率 | 100 Hz |
| 标准通道数 | 64 |
| Patch 长度 | 1 秒 |
| Patch 步长 | 1 秒 |
| Token 数量 | 640 |
| Token 长度 | 100 |
| Transformer 输出层 | `output_layer_idx=8` |
| 特征聚合 | `flatten` |
| 分类类别数 | 4 |

### 5.2 Pipeline 原始输入

```text
signal:        [C, T]
channel_names: 长度 C
sample_rate:   原始采样率
unit:          V / mV / uV / µV
```

BNCI 原始采样率为 250 Hz 时，一个真实 10 秒窗口为：

```text
raw_window: [C, 2500]
```

当前 Stage 0.5 必须输入真实 10 秒窗口，不允许将 4 秒数据补零为 10 秒。

### 5.3 50M 预处理输出

```text
signal:             [64, 1000] float32
channel_valid_mask: [64]       float32
```

通用 Pipeline 使用字典传递模型输入：

```python
{
    "signal": signal,
    "channel_valid_mask": channel_valid_mask,
}
```

### 5.4 Token、Backbone 与分类输出

```text
token_inputs:          [B, 640, 100]
token_channel_indices: [B, 640]
token_time_indices:    [B, 640]
token_valid_mask:      [B, 640]

token_embeddings:      [B, 640, 512]
flatten_features:      [B, 327680]
logits:                [B, 4]
probabilities:         [B, 4]
```

---

## 6. 仓库结构

```text
NCC-OI-BCI/
├── configs/
│   ├── day1_bnci_s01.yaml
│   └── stage05_50m_bnci_s01.yaml
│
├── data/
│   └── processed/
│       └── bnci2014_001_s01.h5       # 本地数据，不提交到 Git
│
├── checkpoints/
│   ├── 50m/
│   │   └── model_deploy.pt           # 50M 部署 checkpoint
│   ├── 50m_bnci2014_001_s01_linear_head.pt
│   └── labram-base.pth               # Stage 0 历史基线
│
├── docs/
│   ├── README_50M_MODEL_ADAPTER.md
│   ├── 50m_adapter_interface.md
│   ├── 50m_model_notes.md
│   ├── current_pip_vs_50M.md
│   ├── standard_64_channels.json
│   └── BEGINNER_GUIDE.md
│
├── scripts/
│   ├── prepare_bnci2014_001.py
│   ├── inspect_dataset.py
│   ├── train_50m_population_head.py
│   ├── export_50m_model_package.py
│   ├── test_50m_offline_window.py
│   ├── replay_offline.py
│   ├── run_pipeline.py
│   ├── smoke_test_50m_tokenization.py
│   ├── smoke_test_50m_backbone.py
│   ├── smoke_test_50m_classifier.py
│   ├── smoke_test_50m_adapter.py
│   ├── smoke_test_50m_runtime.py
│   ├── smoke_test_labram.py
│   └── train_linear_probe.py
│
├── src/bci_dayloop/
│   ├── acquisition/                   # 数据源与 Replay Acquirer
│   ├── data/                          # HDF5 数据读取
│   ├── inference/                     # 滑窗推理、运行控制和观测指标
│   ├── models/
│   │   ├── model_50m/                 # 50M 配置、预处理、Backbone、分类头和 Runtime
│   │   ├── labram_linear.py           # Stage 0 LaBraM 基线
│   │   ├── factory.py
│   │   └── runtime_package.py
│   └── utils/
│
├── tests/                              # 单元测试与集成测试
├── web/
│   ├── app.py
│   └── ui_runtime.py
├── runs/                               # 本地运行结果，不提交到 Git
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## 7. 环境安装

项目使用 Python 3.11。Python 包名为 `bci-dayloop`，导入名为 `bci_dayloop`。

### 7.1 克隆仓库

```bash
git clone https://github.com/ttkx-web/NCC-OI-BCI.git
cd NCC-OI-BCI
```

### 7.2 NVIDIA CUDA 环境

`environment.yml` 只声明可移植的 Python 数值环境；PyTorch GPU wheel 与
CUDA 选择必须按目标平台单独安装，见
[`server_deployment.md`](server_deployment.md)：

```bash
conda env create -f environment.yml
conda activate bci-dayloop
# Install the platform-approved PyTorch wheel first; then:
python -m pip install -e . --no-deps
```

### 7.3 macOS 或 CPU 环境

macOS 或 CPU 环境不需要 GPU PyTorch wheel。可单独创建环境后安装当前项目：

```bash
conda create -n bci-dayloop python=3.11 -y
conda activate bci-dayloop
python -m pip install --upgrade pip
python -m pip install -e .
```

确认导入的是当前仓库：

```bash
python -c "import bci_dayloop; print(bci_dayloop.__file__)"
```

正常情况下，输出路径应指向当前仓库中的：

```text
NCC-OI-BCI/src/bci_dayloop/__init__.py
```

---

## 8. 准备外部文件

模型权重、分类头、HDF5 数据和运行结果不会提交到 Git。运行 Stage 0.5 至少需要：

```text
data/processed/bnci2014_001_s01.h5
checkpoints/50m/model_deploy.pt
checkpoints/50m_bnci2014_001_s01_linear_head.pt
```

### 8.1 50M Backbone

```text
checkpoints/50m/model_deploy.pt
```

该文件必须是 dependency-free 部署 checkpoint，只包含 Tensor 和普通 Python 数据结构。若原始预训练 checkpoint 中包含自定义 `Config` 对象，应先在原 50M 仓库环境中转换为部署 checkpoint。

### 8.2 正式运动想象分类头

```text
checkpoints/50m_bnci2014_001_s01_linear_head.pt
```

分类头 checkpoint 应包含：

```text
format_version
head_state_dict
metadata
```

### 8.3 BNCI HDF5 数据

```text
data/processed/bnci2014_001_s01.h5
```

该文件包含：

```text
0train    # 训练 Session
1test     # 独立测试与 Replay Session
```

---

## 9. 准备 BNCI2014_001 数据

主 Runtime 不依赖 MOABB。MOABB 只在首次下载或重新生成 HDF5 时使用，建议放在独立的数据准备环境中。

### 9.1 创建数据准备环境

```bash
conda create -n bci-dayloop-data python=3.11 -y
conda activate bci-dayloop-data
python -m pip install -r requirements-data.txt
```

### 9.2 下载并生成 HDF5

macOS / Linux：

```bash
export MNE_DATASETS_BNCI_PATH="$PWD/data/moabb_cache"
mkdir -p data/moabb_cache
python scripts/prepare_bnci2014_001.py \
  --config configs/day1_bnci_s01.yaml
```

Windows PowerShell：

```powershell
$env:MNE_DATASETS_BNCI_PATH = "$PWD\data\moabb_cache"
New-Item -ItemType Directory -Force data\moabb_cache | Out-Null
python scripts\prepare_bnci2014_001.py `
  --config configs\day1_bnci_s01.yaml
```

检查数据：

```bash
python scripts/inspect_dataset.py \
  data/processed/bnci2014_001_s01.h5
```

完成后切回主环境：

```bash
conda deactivate
conda activate bci-dayloop
```

---

## 10. 训练正式 50M 线性分类头

仓库已经定义了正式分类头路径。仅在需要重新训练分类头时执行：

```bash
PYTHONPATH=src python -m bci_dayloop.training.model_50m.linear_head \
  --data data/processed/bnci2014_001_s01.h5 \
  --train-session 0train \
  --test-session 1test \
  --checkpoint checkpoints/50m/model_deploy.pt \
  --output checkpoints/50m_bnci2014_001_s01_linear_head.pt \
  --device cpu \
  --window-sec 10 \
  --window-stride-sec 10 \
  --feature-batch-size 1 \
  --feature-cache-dtype float16 \
  --head-batch-size 32 \
  --epochs 100 \
  --head-lr 0.001 \
  --momentum 0 \
  --weight-decay 0.001 \
  --metric-for-best val_bacc \
  --patience 15
```

训练流程：

1. 在原始 Trial 层面对 `0train` 做训练集和验证集划分；
2. 只在相同标签内部拼接 Trial；
3. 构造临时 10 秒单标签窗口；
4. 冻结 50M Backbone；
5. 缓存分类特征；
6. 只训练线性分类头；
7. 根据验证集 Balanced Accuracy 选择最佳 Epoch；
8. 最后在 `1test` 上评估一次；
9. 保存正式分类头和训练报告。

> 当前分类头是正式训练产物，但使用的是 Stage 0.5 临时数据构造方式：将同标签的原始 4 秒 Trial 拼接后截取 10 秒窗口。它不等同于自然连续采集的 10 秒单 Trial 实验。

---

## 11. 导出 Runtime Model Package

Runtime Model Package 是部署时使用的标准模型包。它将模型配置、预处理配置、分类头、类别映射、命令映射和模型文件校验信息组织成独立目录，使 CLI Replay 和 Streamlit 不依赖训练进程中的内存对象。

使用默认路径导出：

```bash
python scripts/export_50m_model_package.py \
  --overwrite
```

完整命令：

```bash
python scripts/export_50m_model_package.py \
  --data data/processed/bnci2014_001_s01.h5 \
  --checkpoint checkpoints/50m/model_deploy.pt \
  --classifier checkpoints/50m_bnci2014_001_s01_linear_head.pt \
  --output runs/stage05_50m/model_package \
  --device cpu \
  --session 1test \
  --step-sec 0.5 \
  --overwrite
```

导出脚本会：

- 加载正式分类头；
- 创建 50M Runtime；
- 保存 Runtime Model Package；
- 保留源分类头训练 metadata；
- 写入 Backbone 和分类头 SHA-256；
- 将 Backbone 路径保存为相对路径；
- 写入 `step_sec=0.5`；
- 使用真实 10 秒 BNCI 窗口执行导出后 Smoke Test；
- 生成 `export_manifest.json`；
- 验证成功后原子替换旧 Model Package。

只导出、不执行 Smoke Test：

```bash
python scripts/export_50m_model_package.py \
  --skip-smoke-test
```

---

## 12. Runtime Model Package 结构

```text
runs/stage05_50m/model_package/
├── model.yaml
├── preprocessing.yaml
├── classifier.pt
├── label_map.json
├── command_map.json
├── base_model.json
└── export_manifest.json
```

### `model.yaml`

保存：

- 模型名称与类别数；
- 类别顺序；
- 10 秒窗口；
- 100 Hz 采样率；
- Patch 配置；
- Transformer 配置；
- `output_layer_idx=8`；
- `aggregation=flatten`；
- `step_sec=0.5`；
- 数据集和任务说明。

### `preprocessing.yaml`

保存：

- 通道和单位配置；
- 带通滤波配置；
- 平均参考配置；
- Z-score 配置；
- 缺失通道策略；
- 严格窗口长度检查。

### `classifier.pt`

保存正式训练后的线性分类头及其训练 metadata。

### `label_map.json`

保存稳定的数字类别顺序，必须与 HDF5 metadata 一致。

### `command_map.json`

默认运动想象命令映射：

```json
{
  "left_hand": "LEFT",
  "right_hand": "RIGHT",
  "feet": "FORWARD",
  "tongue": "STOP"
}
```

### `base_model.json`

保存：

- Backbone checkpoint 相对路径；
- Backbone SHA-256；
- 分类头源文件及 SHA-256；
- 模型类型和加载信息；
- `is_test_head: false`；
- `trained_head: true`。

### `export_manifest.json`

保存导出时间、源文件、输出文件和导出后验证信息。

---

## 13. 离线单窗口验证

使用真实 BNCI 10 秒窗口验证预处理、Backbone、分类头和原始接口一致性：

```bash
python scripts/test_50m_offline_window.py \
  --data data/processed/bnci2014_001_s01.h5 \
  --session 1test \
  --checkpoint checkpoints/50m/model_deploy.pt \
  --classifier checkpoints/50m_bnci2014_001_s01_linear_head.pt \
  --device cpu \
  --window-sec 10 \
  --repeat 2 \
  --json-output runs/stage05_50m/trained_head_offline_report.json
```

应重点确认：

```text
预处理输出：     [64, 1000]
Token 输入：     [1, 640, 100]
Backbone 输出：  [1, 640, 512]
分类特征：       [1, 327680]
类别概率：       [1, 4]
概率和：         约等于 1
重复推理：       输出一致
接口一致性：     通过
```

---

## 14. CLI 伪实时回放

CLI Replay 将 HDF5 中的离线 EEG 拼接为连续数据流，按照指定窗口、步长和回放速度持续执行预处理与推理。

### 14.1 快速检查一个窗口

```bash
python scripts/replay_offline.py \
  --config configs/stage05_50m_bnci_s01.yaml \
  --model-package runs/stage05_50m/model_package \
  --device cpu \
  --max-windows 1 \
  --replay-speed 100
```

### 14.2 按真实速度回放多个窗口

```bash
python scripts/replay_offline.py \
  --config configs/stage05_50m_bnci_s01.yaml \
  --model-package runs/stage05_50m/model_package \
  --device cpu \
  --max-windows 20 \
  --replay-speed 1
```

重点检查：

```text
emitted_windows == target_windows
successful_windows == emitted_windows
failed_windows == 0
last_error_type == null
```

默认运行输出：

```text
runs/stage05_50m/pipeline_windows.jsonl
runs/stage05_50m/replay_summary.json
```

其中：

- JSONL 文件逐窗口记录时间、预测、置信度、延迟和异常信息；
- Summary JSON 汇总窗口完成率、失败窗口、运行时长和延迟统计。

---

## 15. Streamlit 可视化

启动界面：

```bash
streamlit run web/app.py
```

Windows 也可以运行：

```text
run_web.bat
```

推荐选择：

```text
Data:
data/processed/bnci2014_001_s01.h5

Model package:
runs/stage05_50m/model_package

Compute device:
cpu

Session:
1test

Window:
10.0 s

Step:
0.5 s
```

页面当前支持：

- 数据文件和 Model Package 发现；
- 模型 Adapter 与 Acquirer 选择；
- Compute device；
- Session；
- Replay speed；
- Maximum windows；
- Confidence threshold；
- Start / Stop / Restart；
- EEG 波形；
- 当前预测与置信度；
- Prediction History；
- 当前、平均和 P95 延迟；
- 窗口处理和运行状态。

---

## 16. 分层 Smoke Test

```bash
python scripts/smoke_test_50m_tokenization.py
python scripts/smoke_test_50m_backbone.py
python scripts/smoke_test_50m_classifier.py
python scripts/smoke_test_50m_adapter.py
python scripts/smoke_test_50m_runtime.py
```

Stage 0 LaBraM 回归测试：

```bash
python scripts/smoke_test_labram.py \
  --checkpoint checkpoints/labram-base.pth \
  --device cpu
```

仅验证 Pipeline 结构时，可以使用随机初始化：

```bash
python scripts/smoke_test_labram.py \
  --device cpu \
  --random-init
```

---

## 17. 运行测试

编译检查：

```bash
python -m compileall src tests scripts web
```

运行测试：

```bash
pytest
```

测试代码主要覆盖：

- BNCI 窗口构造；
- HDF5 数据读取；
- 模型接口；
- 50M Adapter 接口；
- Runtime Model Package；
- 滑窗推理；
- Replay Acquirer；
- CLI Replay；
- Observability；
- Runtime Control；
- Run Report；
- Streamlit Runtime。

---

## 18. 预处理流程

当前 Stage 0.5 暂定预处理流程：

```text
输入单位统一为 µV
-> 映射到标准 64 通道
-> 缺失通道补零并生成 channel_valid_mask
-> 可选平均参考，默认关闭
-> 在原始采样率下执行 0.1–75 Hz 带通滤波
-> 重采样到 100 Hz
-> 严格检查真实 10 秒 / 1000 点
-> 对有效通道按时间维执行 Z-score
-> 划分为 640 个长度为 100 的 Token
```

标准 64 通道列表见：

```text
docs/standard_64_channels.json
```

---

## 19. 运行指标

### 19.1 Pipeline 指标

- 运行时长；
- Replay speed；
- 预期窗口数；
- 实际发出窗口数；
- 成功推理窗口数；
- 失败窗口数；
- 窗口完成率；
- 异常和超时信息。

### 19.2 延迟指标

- 当前延迟；
- 平均延迟；
- P95 延迟；
- 单窗口推理时间；
- 运行过程中延迟变化。

### 19.3 操作与可观测性

- Start / Stop / Restart 状态；
- 配置加载结果；
- 当前 Pipeline State；
- 预测、置信度和历史记录；
- JSONL 窗口日志；
- Summary JSON；
- 最后一次错误类型与错误信息。

---

## 20. Stage 0 LaBraM 历史基线

Stage 0 使用：

```text
configs/day1_bnci_s01.yaml
```

主要流程：

```text
BNCI2014_001 Subject 1
-> LaBraM Base
-> 缓存 Embedding
-> 训练 Linear Probe
-> 保存 Model Package
-> CLI Replay / Streamlit
```

完整运行：

```bash
python scripts/run_pipeline.py \
  --config configs/day1_bnci_s01.yaml
```

分别运行：

```bash
python scripts/train_linear_probe.py \
  --config configs/day1_bnci_s01.yaml

python scripts/replay_offline.py \
  --config configs/day1_bnci_s01.yaml \
  --max-windows 20
```

LaBraM 仅作为历史基线和回归测试保留，当前默认正式模型为 50M。

---

## 21. 当前边界

1. 当前 50M Runtime 使用 10 秒输入配置；
2. 正式分类头的训练窗口由同标签 4 秒 Trial 临时拼接得到；
3. Replay 中的普通 10 秒滑窗可能跨越不同标签 Trial；
4. 因此 Replay 单窗口预测不适合作为严格模型准确率评估；
5. 目标采样率为 100 Hz，最终模型输入不能保留 50 Hz 以上频率成分；
6. 当前 0.1–75 Hz 滤波设置仍需继续与 50M 预训练真实数据管线核对；
7. 模型 checkpoint、分类头、HDF5 数据和运行结果不随 Git 仓库分发；
8. 当前未自动从火山云 TOS 下载 checkpoint；
9. 当前未完成 GPU 峰值显存统计；
10. 当前尚未接入公司自采离线 EEG 数据；
11. 当前尚未接入 EEG 设备真实实时流；
12. 当前尚未完成简单个体化；
13. 当前尚未实现 Rest-Tuning；
14. 当前尚未实现在线自适应；
15. 当前未接入实体终端控制。

---

## 22. 下一阶段

### Stage 1：简单个体化

目标：

- 使用多被试公开运动想象数据训练群体任务模型；
- 留出未参与群体训练的新被试；
- 使用目标被试少量带标签数据重训个人分类头；
- 对比群体模型与个人模型；
- 支持用户模型保存、加载、隔离和切换；
- 比较不同个体化数据量下的 Accuracy / Macro-F1；
- 验证个人模型在伪实时 Pipeline 中稳定运行。

第一版优先采用：

```text
冻结 50M Backbone
+ 使用目标被试少量数据重新训练分类头
```

LoRA、Adapter 或最后几层微调作为后续增强对照。

### Stage 2A：公司自采离线 EEG 数据兼容

目标：

- 完整保存公司设备的原始 EEG、事件、标签和设备元数据；
- 检查通道名称、通道顺序、采样率、单位和参考方式；
- 检查 NaN、缺失数据、异常通道、丢包和事件时间；
- 将自采数据转换为统一中间数据结构；
- 复用 Stage 0.5 的 50M Pipeline Preprocessor；
- 验证自采数据可以正确进入 50M 并完成前向推理；
- 验证离线读取与伪实时回放窗口一致。

Stage 2A 主要验证数据兼容性，不以高准确率为主要验收门槛。

---

## 23. Roadmap

```text
Stage 0
LaBraM 伪实时基础 Pipeline
        |
        v
Stage 0.5
50M 替换、正式分类头、Runtime Model Package、CLI 和 Streamlit
        |
        +-----------------------------+
        |                             |
        v                             v
Stage 1                         Stage 2A
简单个体化                     自采离线数据兼容
        |                             |
        |                             v
        |                         Stage 2B
        |                         EEG 设备实时接入
        +-------------+---------------+
                      |
                      v
                   Stage 3
          实时 Pipeline + 简单个体化
                      |
                      v
                  Stage 3.5
              Rest-Tuning 静息态个体化
                      |
                      v
                   Stage 4
                  实时自适应
```

---

## 24. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/README_50M_MODEL_ADAPTER.md`](docs/README_50M_MODEL_ADAPTER.md) | 50M Adapter、分类头训练、Model Package、CLI 和 Streamlit 详细说明 |
| [`docs/50m_adapter_interface.md`](docs/50m_adapter_interface.md) | 50M Adapter 接口 |
| [`docs/50m_model_notes.md`](docs/50m_model_notes.md) | 50M 模型接入记录 |
| [`docs/current_pip_vs_50M.md`](docs/current_pip_vs_50M.md) | 原 Pipeline 与 50M 输入要求对比 |
| [`docs/standard_64_channels.json`](docs/standard_64_channels.json) | 50M 标准 64 通道顺序 |
| [`docs/BEGINNER_GUIDE.md`](docs/BEGINNER_GUIDE.md) | 新开发者上手说明 |

---

## 25. 开发约定

- 所有命令默认从仓库根目录执行；
- 使用 `python -m pip install -e .` 安装当前仓库；
- 不在业务脚本中直接使用 `from src...`；
- Python 导入统一使用 `bci_dayloop...`；
- 通用 Pipeline 不硬编码模型专属采样率、通道和窗口参数；
- 模型专属逻辑放在对应 Model Adapter 和 Pipeline Preprocessor 中；
- 训练产物必须能够保存，并在新进程中重新加载；
- checkpoint、HDF5、运行日志、访问密钥和本机绝对路径不得提交到 Git；
- 新阶段功能通过独立分支开发，并通过 Pull Request 合入 `main`；
- 合并前至少运行：

```bash
python -m compileall src tests scripts web
pytest
```

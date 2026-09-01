# NCC-OI-BCI

面向非侵入式 EEG BCI 的统一训练、Runtime Package、离线评估、伪实时回放和真实设备实时推理框架。

项目以公司 **50M EEG 基座模型** 为部署主模型，并接入 **LaBraM**、**CBraMod** 两个基线。当前 `main` 已合并三条能力主线：

- 多数据集适配与严格 sequential 的 **NeuroOnline** 评估；
- **Stage 2B** 的 Neuracle/JellyFish 实时窗口、模型策略和端到端延迟基准；
- 一个共享 50M Backbone、三个分类头的 **workload / attention / emotion** 三状态推理服务。

> 本仓库同时包含研究训练与部署工具。正式任务性能使用 trial-aligned / sequential 协议；连续 Replay 和实时 benchmark 用于验证工程链路与延迟，不能替代正式准确率评估。

## 当前能力

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 统一 Runtime | 已实现 | `RawEEGWindow → prepare() → predict_prepared()` 的模型无关接口。 |
| 模型基线 | 已接入 | 50M、LaBraM、CBraMod 均支持训练、Package 与离线评估。 |
| 数据适配 | 已实现 | BNCI2014_001、Workload、SEED 及顺序数据集适配器。 |
| 个体化 | 已实现 | 50M 群体头、个人头、partial fine-tune、LoRA；CBraMod 支持 LOSO / within-subject。 |
| NeuroOnline V1 | 已实现 | 冻结 Backbone，在线更新 Generator + Head；支持 50M、LaBraM、CBraMod。 |
| 三状态服务 | 已实现 | 一个 50M Backbone 同时输出 workload、attention、emotion。 |
| Stage 2B | 已实现 | 真实设备窗口、通道/输入安全 gate、50M/LaBraM/CBraMod 实时 benchmark。 |
| Rest-Tune | 计划中 | 接口预留，算法流程尚未作为正式能力交付。 |

## 两条部署路径

### 单任务 Runtime Package

适用于运动想象、Workload 等单任务模型，以及 Stage 2B 直接 Runtime 推理：

```text
RawEEGWindow
  → RuntimeModel.prepare()
  → PreparedModelInput
  → RuntimeModel.predict_prepared()
  → ModelOutput
```

Package 的 `model.type` 为 `model_50m`、`labram` 或 `cbramod`，统一用：

```python
from bci_dayloop.packages import load_runtime_package

loaded = load_runtime_package("model_packages/<single-task-package>", device="cpu")
output = loaded.runtime_model.predict(raw_window)
```

### 三状态 HTTP 推理服务

三状态 Package 的 `model.type` 为 `model_50m_multi_head`。它使用一次共享 Backbone forward，再由三个 Head 分别输出 workload、attention、emotion。

```text
采集端 / Rust GUI：解码、切窗、转换为 uV、HTTP 请求
        ↓
Python service：通道适配、重采样、滤波、归一化、共享 50M Backbone、三 Head 推理
```

三状态 Package 使用 `load_inference_package()`，**不**接入 Stage 2B 的单任务 `RealtimeModelPolicyRegistry`。这两条路径目前有意保持独立。

## 仓库结构

```text
NCC-OI-BCI/
├── configs/                 # 训练、Replay、三状态导出、实时 benchmark 配置
├── data/                    # 处理后的 HDF5；大部分数据不提交 Git
├── checkpoints/             # Backbone 与分类头权重（不提交 Git）
├── model_packages/          # 可迁移 Runtime Package（不提交 Git）
├── runs/                    # 训练、评估、Replay、benchmark 输出（不提交 Git）
├── scripts/                 # 命令行入口
├── src/bci_dayloop/         # 可安装的核心实现
├── tests/
└── docs/
```

## 环境与验收

项目要求 Python 3.11。日常本地开发可使用：

```bash
git clone https://github.com/ttkx-web/NCC-OI-BCI.git
cd NCC-OI-BCI

conda create -n bci-dayloop python=3.11 -y
conda activate bci-dayloop
python -m pip install --upgrade pip
python -m pip install -r requirements.local.txt
python -m pip install -e . --no-deps
```

在 GPU 服务器上，请先安装该机器批准的 CUDA PyTorch wheel，再执行 `python -m pip install -e . --no-deps`，避免安装过程覆盖已验证的 CUDA 组合。部署前运行：

```bash
python scripts/check_runtime_environment.py --require-cuda
python -m compileall src scripts tests
PYTEST_TEMP="$(mktemp -d)"
python -m pytest -q --basetemp="$PYTEST_TEMP"
```

服务器部署与 Package 搬运要求见 [docs/server_deployment.md](docs/server_deployment.md)。

## 数据与评估原则

- 训练、验证、最终测试的标签语义必须来自同一数据集元数据；不得按 split 分别重映射。
- 群体模型使用 LOSO：目标被试不参与训练、验证、early stopping 或超参数选择。
- within-subject 仅在源 session 内做分层 trial split，最终 test session 保持隔离。
- 正式性能优先使用 `evaluate_runtime_package.py` 的 trial-aligned 评估，报告 Accuracy、Balanced Accuracy、Macro-F1 和混淆矩阵。
- NeuroOnline 使用严格按 trial 因果顺序的 sequential 协议；`random_permutation` 仅用于诊断。
- 每个正式 Package、训练运行和评估输出都应保存 class semantics、preprocessing / checkpoint hash、split 和配置。

数据适配与 CBRaMod 基线细节见 [docs/cbramod_baseline_protocol.md](docs/cbramod_baseline_protocol.md)。

## 常用工作流

所有脚本都支持 `--help`。下面的 `<...>` 是需要按本机实际路径替换的占位符。

### 1. 准备和检查数据

```bash
# BNCI2014_001
python scripts/prepare_bnci2014_001.py --help
python scripts/inspect_dataset.py data/processed/bnci2014_001/subject_01.h5

# 其他已支持数据集
python scripts/prepare_workload_hdf5.py --help
python scripts/prepare_seed_preprocessed_eeg_hdf5.py --help
python scripts/prepare_mema_for_dl_hdf5.py --help
```

顺序评估和跨数据集训练会经由 `SequentialDataset` / dataset adapter 加载数据；不要为新数据集在训练脚本中临时复制 HDF5 读取逻辑。

### 2. 训练单任务分类头

```bash
# 50M：群体 LOSO 分类头 / 兼容 partial 与 LoRA 适配模式
python scripts/train_50m_population_head.py --help

# 50M：目标被试个人分类头
python scripts/train_50m_personal_head.py --help

# LaBraM：群体 LOSO 分类头
python scripts/train_labram_population_head.py --help

# CBRaMod：LOSO 或 within-subject；可选择 strict22 或 Neuracle live19 profile
python scripts/train_cbramod_population_head.py --help
```

一个 50M 群体头的最小示例：

```bash
python scripts/train_50m_population_head.py \
  --data-root data/processed/bnci2014_001 \
  --data-pattern "subject_{subject:02d}.h5" \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subject 1 \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --window-sec 4 \
  --device cuda
```

训练产物必须保留训练报告和 checkpoint metadata；不要把真实数据、模型权重或 `runs/` 产物提交到 Git。

### 3. 导出并评估单任务 Runtime Package

```bash
# 导出；各脚本会验证 Package 能重新加载并进行 smoke test
python scripts/export_50m_model_package.py --help
python scripts/export_labram_model_package.py --help
python scripts/export_cbramod_model_package.py --help

# 使用 trial-aligned 协议评估正式单任务 Package
python scripts/evaluate_runtime_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package model_packages/<single-task-package> \
  --session 1test \
  --device cuda \
  --output runs/evaluation/<name>.json
```

单任务 schema-v2 Package 典型结构：

```text
<package>/
├── package.yaml
├── backbone.pt
├── classifier.pt
├── preprocessing.yaml
└── metrics.json
```

### 4. Replay 与 NeuroOnline

```bash
# 连续滑窗 Replay；用于链路和延迟验证
python scripts/replay_offline.py \
  --config configs/stage1/replay_population_4s.yaml \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package model_packages/<single-task-package> \
  --online-strategy none \
  --device cuda

# 严格 sequential：对比静态模型和 NeuroOnline
python scripts/evaluate_neuroonline_sequential.py \
  --data data/processed/<dataset-or-subject>.h5 \
  --model-package model_packages/<single-task-package> \
  --session <session> \
  --online-strategy both \
  --device cuda \
  --output-dir runs/neuroonline/<experiment>
```

`replay_offline.py` 不是 NeuroOnline 的正式准确率协议；正式对比请使用 `evaluate_neuroonline_sequential.py`。

### 5. 三状态 Package 与服务

先在 [`configs/three_mental_states/export.yaml`](configs/three_mental_states/export.yaml) 填写 Backbone、三个 Head 和输出目录，再导出：

```bash
python scripts/export_50m_multi_head_model_package.py

python scripts/serve_inference.py \
  --model-package model_packages/50m_three_mental_states \
  --host 127.0.0.1 \
  --port 8767 \
  --device cuda
```

验证 direct / Package / decoder / HTTP 一致性：

```bash
python scripts/verify_three_state_inference.py \
  --mode all \
  --model-package model_packages/50m_three_mental_states \
  --input-h5 data/processed/<input>.h5 \
  --device cuda
```

三状态 HTTP 请求、响应和 Rust 客户端边界见 [docs/inference_service.md](docs/inference_service.md)。

### 6. Stage 2B 真实设备基准

Stage 2B 对单任务 50M、LaBraM、CBraMod Package 做真实来源窗口与延迟测试。设备地址只通过环境变量提供，不能写入配置、日志或提交：

```bash
export NEURACLE_JELLYFISH_HOST='<jellyfish-host>'

python scripts/probe_neuracle_runtime_inference.py --help

python scripts/run_device_window_latency_benchmark.py \
  --config configs/benchmarks/window_latency_live_1_2_3_4s.yaml \
  --device cuda
```

实时路径会对来源、通道、单位、窗口和 Runtime Package 契约执行 fail-closed 验证。不同窗口长度必须使用与之匹配的 Package；不能将任意 4 秒 Package 直接用于 1/2/3 秒测试。

## 文档索引

| 主题 | 文档 |
|---|---|
| 三状态服务与 HTTP 契约 | [docs/inference_service.md](docs/inference_service.md) |
| Stage 2B 入口与基准说明 | [docs/README_stage2b.md](docs/README_stage2b.md) |
| Ubuntu / CUDA 部署验收 | [docs/server_deployment.md](docs/server_deployment.md) |
| CBRaMod 冻结骨干基线 | [docs/cbramod_baseline_protocol.md](docs/cbramod_baseline_protocol.md) |
| 50M Runtime 接口 | [docs/README_50M_MODEL_ADAPTER.md](docs/README_50M_MODEL_ADAPTER.md) |
| 历史 Stage 0 / 0.5 说明 | [docs/README_stage0.md](docs/README_stage0.md)、[docs/README_stage05.md](docs/README_stage05.md) |

## 贡献与 Git 约定

- 新模型或新数据集先实现/注册适配器，再接入 trainer、Package 和评估，不要在已有脚本中硬编码特例。
- 修改 Runtime 输入、Package schema、实时 policy 或数据 split 时，必须同步补测试和文档。
- `data/raw/`、大部分 `data/processed/`、`checkpoints/`、`model_packages/`、`runs/` 均为本地资产；仅已明确允许的 dataset manifest 可提交。
- 对真实设备的地址、账号、密码、序列号和受试者信息一律不提交。

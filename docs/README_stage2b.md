# NCC-OI-BCI

基于公司 **50M EEG 基座模型** 的离线训练、个体化适配、伪实时回放与 Streamlit 演示基础设施。

当前主线任务是 **BNCI2014_001 四分类运动想象**，并以统一的 Model Adapter、Runtime Model Package 和 Pipeline 接口，逐步扩展到公司自采 EEG、真实设备接入、Rest-Tuning 与在线自适应。

> 当前状态：Stage 0、Stage 0.5 已完成；Stage 1A（冻结 50M Backbone、群体头与个人头、个人模型包、Registry、双模型对比）已实现。正式交付前需完成本文末尾的本地验收。

---

## 1. 当前任务

| EEG 类别 | 控制命令 |
|---|---|
| `left_hand` | `LEFT` |
| `right_hand` | `RIGHT` |
| `feet` | `FORWARD` |
| `tongue` | `STOP` |

当前默认模型是公司 50M EEG 基座模型。Stage 0 使用的 LaBraM 作为历史基线保留，用于回归测试和模型替换前后的工程对照。

---

## 2. 阶段进度

| 阶段 | 目标 | 状态 |
|---|---|---|
| Stage 0 | LaBraM 伪实时基础 Pipeline | 已完成 |
| Stage 0.5 | 50M 替换、正式分类头、Runtime Package、CLI、Streamlit | 已完成 |
| Stage 1A | 冻结 50M Backbone 的群体头与个人头适配 | 已实现，待最终验收 |
| Stage 1B | 解冻最后若干 Backbone Block 的任务相关个体化 | 计划中 |
| Stage 2A | 公司自采离线 EEG 数据兼容 | 推进中 |
| Stage 2B | EEG 设备真实实时接入 | 计划中 |
| Stage 3 | 实时 Pipeline 与简单个体化合并 | 计划中 |
| Stage 3.5 | 基于静息态 EEG 的 Rest-Tuning | 计划中 |
| Stage 4 | 实时在线自适应 | 计划中 |

---

## 3. 系统架构

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
Window Buffer / Trial-aligned Window
        |
        v
Model-specific Preprocessor
├── 50M Preprocessor
└── LaBraM Preprocessor
        |
        v
Model Adapter
├── 50M Adapter
└── LaBraM Adapter
        |
        v
Runtime Model Package
        |
        v
SlidingWindowDecoder / Comparison Replay
        |
        v
PipelineController
├── CLI Replay
├── Streamlit UI
├── JSONL Window Log
└── Summary / Latency Metrics
```

通用 Pipeline 不应硬编码某个模型的窗口长度、采样率、通道顺序或分类头维度。这些约束由模型配置、Adapter 和 Runtime Model Package 共同提供。

---

## 4. 当前 4 秒 50M Runtime 契约

Stage 1A 使用 BNCI2014_001 的真实 4 秒 Trial，不再将 4 秒 Trial 补零或默认拼接成 10 秒输入。

| 配置项 | 当前值 |
|---|---|
| 数据集 | BNCI2014_001 |
| 任务 | 四分类运动想象 |
| 原始采样率 | 250 Hz |
| 原始 Trial | 4 秒，约 `[C, 1000]` |
| 目标采样率 | 100 Hz |
| 标准通道数 | 64 |
| 预处理输出 | `[64, 400]` |
| Patch 长度 | 1 秒 |
| Patch 步长 | 1 秒 |
| 输入时间 Patch 数 | 4 |
| Token 数 | `64 × 4 = 256` |
| Token 长度 | 100 |
| Token Embedding | `[B, 256, 512]` |
| Flatten 特征维度 | `256 × 512 = 131072` |
| 预训练时间位置数 | `model_n_time_patches=10` |
| Transformer 输出层 | `output_layer_idx=8` |
| 默认聚合 | `flatten` |
| 分类类别数 | 4 |
| Runtime 滑窗步长 | 0.5 秒 |

### 为什么 4 秒输入仍保留 10 个模型时间位置

50M Backbone 预训练时使用 10 秒输入，对应 10 个时间位置。Stage 1A 的下游输入只包含前 4 个实际时间 Patch，但模型仍保留 10 个预训练时间位置参数：

```text
实际输入时间 Patch：4
模型时间位置容量：10
实际 Token 数：64 × 4 = 256
```

因此，`model_n_time_patches=10` 不代表把输入补成 10 秒。

---

## 5. Stage 1A 实验协议

### 5.1 群体头

以 Subject 1 为目标被试时：

```text
Population Train:
Subject 2–9 / 0train

Population Validation:
Subject 2–9 / 1test

Target Final Test:
Subject 1 / 1test
```

目标被试不得进入群体头训练或验证。最终测试集只用于训练完成后的报告，不参与模型选择。

### 5.2 个人头

```text
Personalization Source:
Target Subject / 0train

Personal Train:
N Trials per Class

Personal Validation:
固定且与训练互斥的 Trials per Class

Final Test:
Target Subject / 1test
```

规则：

1. 先在源 Trial 层面完成训练/验证划分；
2. Personal Validation 使用固定 `validation_seed`；
3. 个人训练预算由 `personalization_seed` 控制；
4. 同一 Seed 下的 5/10/20/40 Trials 预算保持嵌套；
5. 50M Backbone 冻结，只训练个人任务分类头；
6. 是否激活个人模型只根据 Personal Validation；
7. `1test` 不参与个人模型选择。

### 5.3 激活规则

```text
Personal Validation 指标提升达到阈值
→ 注册并设为 Active

未达到阈值
→ 注册为 Candidate
→ 原 Active Model 保持不变
```

默认比较指标为 Balanced Accuracy。可通过：

```bash
--activation-metric balanced_accuracy
--min-personal-val-gain 0.02
```

要求个人验证集 BAcc 至少提升 2 个百分点。

---

## 6. 目录规范

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
│           ├── subject_01.h5
│           ├── subject_02.h5
│           └── ...
│
├── checkpoints/
│   ├── backbones/
│   │   ├── 50m/
│   │   │   └── model_deploy.pt
│   │   └── labram/
│   │       └── labram_base.pth
│   │
│   └── heads/
│       ├── stage05/
│       └── stage1/
│           └── bnci2014_001/
│               └── subject_01/
│                   ├── population/
│                   │   └── 4s_flatten/
│                   │       └── head.pt
│                   └── personal/
│                       └── 4s_flatten/
│                           └── trials_20/
│                               └── seed_42/
│                                   └── head.pt
│
├── runs/
│   └── stage1/
│       └── bnci2014_001/
│           └── subject_01/
│               ├── population/
│               │   └── 4s_flatten/
│               │       └── <timestamp>/
│               ├── personal/
│               │   └── 4s_flatten/
│               │       └── trials_20/
│               │           └── seed_42/
│               │               └── <timestamp>/
│               └── comparisons/
│                   └── 1test/
│                       └── <timestamp>/
│
├── model_packages/
│   └── stage1/
│       └── bnci2014_001/
│           └── subject_01/
│               ├── population/
│               │   └── 4s_flatten/
│               │       └── <version>/
│               └── personal/
│                   └── 4s_flatten/
│                       └── trials_20/
│                           └── seed_42/
│                               └── <version>/
│
├── registries/
│   └── stage1_personal_models.json
│
├── scripts/
├── src/bci_dayloop/
├── tests/
├── web/
└── README.md
```

目录职责：

| 目录 | 内容 |
|---|---|
| `data/` | 原始数据、缓存和处理后的 HDF5 |
| `checkpoints/backbones/` | 基座模型权重 |
| `checkpoints/heads/` | 稳定保存的群体头与个人头 |
| `runs/` | 每次实验的特征、日志、指标和报告 |
| `model_packages/` | CLI 与 Streamlit 可直接加载的模型包 |
| `registries/` | Active/Candidate 个人模型索引 |

代码中的路径优先通过 `src/bci_dayloop/utils/paths.py` 生成。报告、Registry 和配置尽量避免保存开发者电脑的绝对路径。

---

## 7. 环境安装

项目使用 Python 3.11。Python 包名为 `bci-dayloop`，导入名为 `bci_dayloop`。

### 7.1 克隆分支

```bash
git clone https://github.com/ttkx-web/NCC-OI-BCI.git
cd NCC-OI-BCI
git checkout feat/downstream-4s
```

### 7.2 NVIDIA CUDA 环境

```bash
conda env create -f environment.yml
conda activate bci-dayloop
# Install the platform-approved PyTorch wheel first; then:
python -m pip install -e . --no-deps
```

### 7.3 macOS / CPU 环境

macOS 不应安装 CUDA 依赖。建议：

```bash
conda create -n bci-dayloop python=3.11 -y
conda activate bci-dayloop
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

检查当前 Python 和项目路径：

```bash
which python
python -c "import sys, bci_dayloop; print(sys.executable); print(bci_dayloop.__file__)"
```

测试时使用：

```bash
python -m pytest
```

避免终端中的 `pytest` 来自其他 Conda 环境。

---

## 8. 准备外部文件

下列文件默认不提交到 Git：

```text
data/processed/bnci2014_001/subject_01.h5
...
data/processed/bnci2014_001/subject_09.h5

checkpoints/backbones/50m/model_deploy.pt
checkpoints/backbones/labram/labram_base.pth
```

Stage 1A 的分类头由训练脚本生成：

```text
checkpoints/heads/stage1/bnci2014_001/subject_01/population/4s_flatten/head.pt

checkpoints/heads/stage1/bnci2014_001/subject_01/personal/4s_flatten/trials_20/seed_42/head.pt
```

检查数据：

```bash
python scripts/inspect_dataset.py \
  data/processed/bnci2014_001/subject_01.h5
```

---

## 9. 训练群体头

下面示例以 Subject 1 为目标被试：

```bash
python scripts/train_50m_population_head.py \
  --data-root data/processed/bnci2014_001 \
  --target-subject 1 \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --window-sec 4 \
  --window-stride-sec 4 \
  --window-construction direct_trial \
  --model-n-time-patches 10 \
  --target-sample-rate 100 \
  --patch-sec 1 \
  --patch-stride-sec 1 \
  --output-layer-idx 8 \
  --aggregation flatten \
  --device cpu \
  --head-device cpu \
  --feature-batch-size 1 \
  --feature-cache-dtype float16 \
  --head-batch-size 32 \
  --epochs 100 \
  --head-lr 0.001 \
  --weight-decay 0.001 \
  --metric-for-best val_bacc \
  --patience 15
```

默认分类头：

```text
checkpoints/heads/stage1/bnci2014_001/subject_01/population/4s_flatten/head.pt
```

默认 Run 目录：

```text
runs/stage1/bnci2014_001/subject_01/population/4s_flatten/<timestamp>/
```

主要输出：

```text
epoch_metrics.csv
population_training_report.json
run_config.json
summary.json
features_*.pt                 # 仅在启用缓存保存时
```

---

## 10. 导出群体 Runtime Model Package

群体头训练完成后，需要单独导出群体 Runtime Package：

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

导出脚本会从分类头 metadata 恢复实际 Runtime 契约，并在保存前后分别执行一次加载与推理 Smoke Test。

标准 Runtime Package 包含：

```text
model.yaml
preprocessing.yaml
classifier.pt
label_map.json
command_map.json
base_model.json
export_manifest.json
```

`base_model.json` 使用相对于最终 Package 目录的 Backbone 引用，避免绑定开发者电脑的绝对路径。

---

## 11. 训练个人头、打包并注册

```bash
python scripts/train_50m_personal_head.py \
  --data-root data/processed/bnci2014_001 \
  --target-subject 1 \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --population-head checkpoints/heads/stage1/bnci2014_001/subject_01/population/4s_flatten/head.pt \
  --window-sec 4 \
  --window-stride-sec 4 \
  --window-construction direct_trial \
  --model-n-time-patches 10 \
  --aggregation flatten \
  --device cpu \
  --head-device cpu \
  --trials-per-class 20 \
  --validation-trials-per-class 16 \
  --validation-seed 2026 \
  --personalization-seed 42 \
  --seed 42 \
  --window-seed 42 \
  --head-init population \
  --feature-batch-size 1 \
  --feature-cache-dtype float16 \
  --head-batch-size 8 \
  --optimizer adamw \
  --epochs 50 \
  --head-lr 0.0001 \
  --weight-decay 0.05 \
  --metric-for-best val_bacc \
  --patience 8 \
  --scheduler plateau \
  --scheduler-factor 0.3 \
  --scheduler-patience 3 \
  --activation-metric balanced_accuracy \
  --min-personal-val-gain 0.02
```

脚本执行顺序：

```text
读取并验证群体头
→ 划分 Personal Train / Validation
→ 冻结 50M Backbone 并提取特征
→ 训练个人分类头
→ 通过 Personal Validation 选择最佳 Epoch
→ 保存并重新加载验证个人头
→ 自动导出个人 Runtime Package
→ 创建带训练和指标信息的个人模型包
→ 注册到 Registry
→ 根据 Personal Validation 决定是否设为 Active
→ 最后在 1test 上生成独立报告
```

默认个人头：

```text
checkpoints/heads/stage1/bnci2014_001/subject_01/personal/4s_flatten/trials_20/seed_42/head.pt
```

正式个人 Package：

```text
model_packages/stage1/bnci2014_001/subject_01/personal/4s_flatten/trials_20/seed_42/<timestamp>/
```

Registry：

```text
registries/stage1_personal_models.json
```

个人 Package 在标准 Runtime 文件外增加：

```text
personalization.json
training.json
metrics.json
```

---

## 12. 查询 Active 个人模型

```bash
python - <<'PY'
from bci_dayloop.personalization import PersonalModelRegistry
from bci_dayloop.utils.paths import personal_registry_path

registry = PersonalModelRegistry(
    personal_registry_path("stage1")
)

path = registry.resolve_active_runtime(
    user_id="subject_01",
    task="motor_imagery_4class",
)

print(path)
PY
```

Registry 中保存相对于 Registry 目录的 Package 路径；移动整个项目目录后仍可解析。

---

## 13. 群体模型与个人模型对比

`replay_compare_population_personal.py` 将同一个标签明确的 EEG Trial 分别送入群体模型和个人模型。

```bash
python scripts/replay_compare_population_personal.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --session 1test \
  --population-package <population-package-path> \
  --personal-package <personal-package-path> \
  --device cpu \
  --window-mode direct_trial \
  --max-windows 20 \
  --alternate-model-order
```

逐窗口记录：

```text
Ground Truth
Population Prediction / Confidence
Personal Prediction / Confidence
Models Agree
Population Correct
Personal Correct
Shared Preprocessing Latency
Population / Personal Model Latency
Source Trial IDs
```

汇总指标：

```text
Accuracy
Balanced Accuracy
Macro-F1
Confusion Matrix
Agreement Rate
Personalization Gain
P50 / P95 Latency
Window Completion Rate
```

`Agreement` 只表示两个模型预测类别是否一致，不等于准确率。

---

## 14. CLI Replay

### 14.1 群体模型

```bash
python scripts/replay_offline.py \
  --config configs/stage1/replay_population_4s.yaml \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package <population-package-path> \
  --device cpu \
  --max-windows 20 \
  --replay-speed 100
```

### 14.2 个人模型

```bash
python scripts/replay_offline.py \
  --config configs/stage1/replay_personal_4s.yaml \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --model-package <personal-package-path> \
  --device cpu \
  --max-windows 20 \
  --replay-speed 100
```

建议在命令行显式传入 `--model-package`，避免配置文件中的示例版本号与本地真实时间戳不一致。

CLI 会检查请求的窗口长度和步长是否与 Runtime Package 一致。

---

## 15. Streamlit

启动：

```bash
streamlit run web/app.py
```

页面支持：

- 递归发现 `model_packages/` 和旧版 `runs/` 下的 Runtime Package；
- 递归发现 `data/processed/` 下的 HDF5；
- 页面只显示相对于项目根目录的路径；
- 从所选 Package 读取窗口长度和步长；
- Start / Stop / Restart；
- EEG 波形；
- 当前预测、置信度和控制命令；
- Prediction History；
- 当前、平均和 P95 延迟；
- 窗口完成率与错误状态；
- 可选 JSONL 日志。

macOS 上选择 `cpu`。当前 CLI Replay 仅接受 `cpu` 或 `cuda`。

---

## 16. 配置文件说明

Stage 1 配置：

```text
configs/stage1/replay_population_4s.yaml
configs/stage1/replay_personal_4s.yaml
```

个人模型的标准路径顺序是：

```text
personal/4s_flatten/trials_20/seed_42/<version>
```

个人分类头的标准路径是：

```text
personal/4s_flatten/trials_20/seed_42/head.pt
```

由于 Package 使用时间戳版本目录，配置文件中的 Package 路径只能作为示例。正式运行优先通过 `--model-package` 传入实际路径，或从 Registry 查询 Active Runtime。

---

## 17. 测试

### 17.1 编译与导入检查

```bash
python -m compileall -q src scripts web tests

python scripts/train_50m_population_head.py --help
python scripts/train_50m_personal_head.py --help
python scripts/export_50m_model_package.py --help
python scripts/replay_compare_population_personal.py --help
python scripts/replay_offline.py --help
```

### 17.2 Stage 1A 专项测试

```bash
python -m pytest -q \
  tests/test_paths.py \
  tests/test_personalization_split.py \
  tests/test_personalization_package.py \
  tests/test_personalization_registry.py \
  tests/test_model_50m_4s_runtime.py \
  tests/test_replay_compare_population_personal.py
```

### 17.3 全量测试

```bash
python -m pytest -q
```

正式交付要求：

```text
0 failed
0 errors
```

---

## 18. 最终交付验收

在合并到 `main` 或创建 Release Tag 前，依次完成：

```text
[ ] python -m compileall -q src scripts web tests
[ ] python -m pytest -q
[ ] 群体头训练脚本 --help 正常
[ ] 个人头训练脚本 --help 正常
[ ] 群体 Runtime Package 可加载
[ ] 个人 Runtime Package 可加载
[ ] 双模型 Compare 至少完成 2 个 direct_trial 窗口
[ ] CLI Replay 至少完成 2 个窗口
[ ] Streamlit 能发现数据和模型包
[ ] Streamlit Start 正常
[ ] Streamlit Stop 正常
[ ] Streamlit Restart 正常
[ ] JSONL 与 Summary JSON 正常生成
[ ] Registry 能解析 Active Runtime
[ ] Git 不包含 HDF5、权重、运行结果或本地绝对路径
```

建议同时执行：

```bash
git grep -nE \
'runs/stage1_4s|runs/stage1/users|checkpoints/50m/|checkpoints/stage1/|bnci2014_001_s01\.h5|Temporary 10-second baseline'
```

主线代码、配置和 README 中不应再残留旧路径或旧 10 秒 Stage 1 描述。

---

## 19. 已知限制

- Stage 1A 只个体化任务分类头，50M Backbone 仍然冻结；
- 当前个人适配需要目标被试少量有标签运动想象数据；
- 尚未实现基于无标签静息态数据的 Rest-Tuning；
- 尚未实现真实设备驱动接入；
- CLI/Streamlit 当前不自动热切换 Registry 中的新 Active Model；
- 连续滑窗 Replay 可能跨原始 Trial，适合工程链路验证；正式准确率比较优先使用 `direct_trial`；
- `flatten` 头输入维度较大，少样本条件下可进一步对比 `mean` 聚合；
- macOS 上宽 Flatten Head 训练优先使用 CPU，避免已知 MPS 稳定性问题。

---

## 20. 常见问题

### Runtime Package 找不到 Backbone

检查：

```text
model_package/base_model.json
```

Backbone 路径应相对于 Package 最终目录解析，并指向：

```text
checkpoints/backbones/50m/model_deploy.pt
```

已经导出的 Package 不建议在不同目录之间直接移动；应在最终目录重新执行导出。

### Streamlit 报 HDF5 file not found

确认：

```bash
ls -lh data/processed/bnci2014_001/subject_01.h5
```

并从仓库根目录启动：

```bash
streamlit run web/app.py
```

### 测试导入了错误的 Conda 环境

检查：

```bash
which python
python -c "import sys; print(sys.executable)"
```

然后使用：

```bash
conda activate bci-dayloop
python -m pytest -q
```

### Git 忽略规则

数据、Checkpoint、运行结果、模型包和 Registry 都是本地产物：

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

仓库只保留各目录的 `.gitkeep` 和代码配置。

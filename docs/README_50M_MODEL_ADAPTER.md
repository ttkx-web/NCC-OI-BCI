# 50M Model Adapter

本目录提供公司 50M EEG 基座模型与 `NCC-OI-BCI` 模型无关 Pipeline 之间的适配、训练、模型包导出和伪实时运行能力。

当前阶段使用：

- BNCI2014_001 Subject 1；
- 50M 原始 10 秒输入配置；
- 100 Hz；
- 标准 64 通道；
- 1 秒 Patch；
- 640 个 Token；
- 第 9 个 Transformer Block 后的表示（`output_layer_idx=8`）；
- Flatten 特征；
- 正式训练后的运动想象线性分类头。

当前正式分类头路径：

```text
checkpoints/50m_bnci2014_001_s01_linear_head.pt
```

> 当前分类头采用阶段 0.5 的临时 10 秒数据构造方式：先按标签分组，再将同标签的原始 4 秒 Trial 拼接并截取 10 秒窗口。它已经是正式训练产物，不再是随机测试头，但仍不等同于自然连续采集的 10 秒单 Trial 实验。

## 1. 当前状态

已实现：

- 50M 部署配置；
- 标准 64 通道映射与通道别名归一化；
- 缺失通道补零及 `channel_valid_mask`；
- 0.1–75 Hz 带通滤波；
- 100 Hz 重采样；
- 10 秒严格窗口检查；
- 有效通道按时间维 Z-score；
- `[64, 1000] → [640, 100]` Token 化；
- 50M Backbone 构建、冻结和部署 checkpoint 加载；
- 指定 Transformer 层的 Token Embedding 提取；
- Flatten / Mean 特征聚合；
- 线性任务分类头训练、保存和加载；
- `predict_proba()`、`predict()` 和 `extract_embeddings()`；
- 字典式 `signal + channel_valid_mask` 通用模型输入；
- Runtime Package 保存和加载；
- CLI Replay 接入；
- Streamlit 接入；
- Start / Stop / Restart；
- 窗口计数、JSONL 日志和 Summary JSON；
- BNCI 离线真实 10 秒窗口测试；
- 正式 Model Package 导出脚本。

暂未实现：

- 火山云 TOS 自动下载；
- GPU 峰值显存统计；
- 4 秒窗口适配及对应正式分类头；
- 真实 EEG 设备接入；
- 个体化微调、Rest-Tuning 和在线自适应。

## 2. 目录结构

```text
src/bci_dayloop/models/model_50m/
├── __init__.py
├── config.py
├── preprocessing.py
├── pipeline_preprocessor.py
├── tokenization.py
├── backbone.py
├── classifier.py
├── adapter.py
└── runtime.py

scripts/
├── train_50m_population_head.py       # Stage-1 population/LOSO trainer
├── export_50m_model_package.py
├── smoke_test_50m_tokenization.py
├── smoke_test_50m_backbone.py
├── smoke_test_50m_classifier.py
├── smoke_test_50m_adapter.py
├── smoke_test_50m_runtime.py
└── test_50m_offline_window.py
```

## 3. 输入输出契约

### 3.1 Pipeline 原始窗口

```text
signal:        [C, T]
channel_names: 长度 C
sample_rate:   原始采样率
unit:          V / mV / uV / µV
```

阶段 0.5 必须输入真实 10 秒窗口，不允许把 4 秒窗口补零成 10 秒。

BNCI 原始采样率为 250 Hz 时：

```text
raw_window: [C, 2500]
```

### 3.2 50M 预处理输出

```text
signal:             [64, 1000] float32
channel_valid_mask: [64]       float32
```

通用 Pipeline 中以字典传递：

```python
{
    "signal": signal,
    "channel_valid_mask": channel_valid_mask,
}
```

### 3.3 Token 输入

```text
token_inputs:          [B, 640, 100] float32
token_channel_indices: [B, 640]      int64
token_time_indices:    [B, 640]      int64
token_valid_mask:      [B, 640]      float32
```

### 3.4 Backbone 和分类输出

```text
token_embeddings: [B, 640, 512]
flatten_features: [B, 327680]
logits:           [B, num_classes]
probabilities:    [B, num_classes]
```

## 4. 环境安装

```bash
conda activate bci-dayloop
python -m pip install -e . --no-deps
```

确认导入的是当前仓库：

```bash
python -c "import bci_dayloop; print(bci_dayloop.__file__)"
```

## 5. 外部文件

模型权重、分类头和 HDF5 数据不提交到 Git。当前运行至少需要：

```text
data/processed/bnci2014_001/subject_01.h5
checkpoints/backbones/50m/model_deploy.pt
checkpoints/50m_bnci2014_001_s01_linear_head.pt
```

### 5.1 50M Backbone

`model.pt` 必须是只包含 Tensor 和普通 Python 数据结构的部署 checkpoint。原始预训练 checkpoint 如果包含 `configs.Config` 对象，应先在原 50M 仓库环境中转换。

### 5.2 正式分类头

```text
checkpoints/50m_bnci2014_001_s01_linear_head.pt
```

该文件应包含：

```text
format_version
head_state_dict
metadata
```

Model Package 导出时会保留分类头中的训练 metadata，并写入源文件 SHA-256。

## 6. 训练正式线性分类头

```bash
PYTHONPATH=src python -m bci_dayloop.training.model_50m.linear_head \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --train-session 0train \
  --test-session 1test \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
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

训练脚本会：

1. 在原始 Trial 层面对 `0train` 做训练/验证划分；
2. 只在相同标签内部拼接 Trial；
3. 构造临时 10 秒单标签窗口；
4. 冻结 50M Backbone；
5. 缓存分类特征；
6. 只训练线性分类头；
7. 根据验证集 Balanced Accuracy 选择最佳 Epoch；
8. 最后在 `1test` 上评估一次；
9. 保存正式分类头和训练报告。

## 7. 导出正式 Runtime Model Package

使用：

```bash
python scripts/export_50m_model_package.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --classifier checkpoints/50m_bnci2014_001_s01_linear_head.pt \
  --output runs/stage05_50m/model_package \
  --device cpu \
  --session 1test \
  --step-sec 0.5 \
  --overwrite
```

导出脚本会：

- 加载正式分类头；
- 创建 Runtime；
- 保存 `classifier.pt`；
- 保留源分类头训练 metadata；
- 将 `is_test_head` 设置为 `false`；
- 写入 Backbone 和分类头 SHA-256；
- 将 Backbone 路径保存为相对路径，避免写死开发者本机绝对路径；
- 写入 `step_sec=0.5`；
- 使用一段真实 10 秒长度的 BNCI 窗口进行导出后 Smoke Test；
- 原子替换旧 Model Package；
- 生成 `export_manifest.json`。

成功后目录为：

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

不希望覆盖现有 Model Package 时，不要传 `--overwrite`。

只导出、不运行 Smoke Test：

```bash
python scripts/export_50m_model_package.py \
  --skip-smoke-test
```

## 8. Model Package 说明

### `model.yaml`

保存：

- `name: 50m-linear`；
- 类别数与类别顺序；
- 10 秒窗口；
- 100 Hz；
- Patch 配置；
- Transformer 配置；
- `output_layer_idx=8`；
- `aggregation=flatten`；
- `step_sec=0.5`；
- 任务和数据集说明。

### `preprocessing.yaml`

保存：

- 带通滤波；
- 平均参考配置；
- Z-score；
- 缺失通道策略；
- 严格窗口长度检查。

### `classifier.pt`

保存正式训练后的线性头，并保留训练 metadata。

### `base_model.json`

保存：

- Backbone checkpoint 相对路径；
- Backbone SHA-256；
- 分类头源文件及 SHA-256；
- `is_test_head: false`；
- `trained_head: true`。

### `label_map.json`

保存稳定的数字类别顺序，必须与 HDF5 metadata 一致。

### `command_map.json`

默认运动想象映射：

```json
{
  "left_hand": "LEFT",
  "right_hand": "RIGHT",
  "feet": "FORWARD",
  "tongue": "STOP"
}
```

可以通过 `--command-map-json` 提供其他映射。

## 9. 逐层测试

```bash
python scripts/smoke_test_50m_tokenization.py
python scripts/smoke_test_50m_backbone.py
python scripts/smoke_test_50m_classifier.py
python scripts/smoke_test_50m_adapter.py
python scripts/smoke_test_50m_runtime.py
```

真实 BNCI 离线窗口测试：

```bash
python scripts/test_50m_offline_window.py \
  --data data/processed/bnci2014_001/subject_01.h5 \
  --session 1test \
  --checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --classifier checkpoints/50m_bnci2014_001_s01_linear_head.pt \
  --device cpu \
  --window-sec 10 \
  --repeat 2 \
  --json-output runs/stage05_50m/trained_head_offline_report.json
```

应确认：

```text
预处理输出：     [64, 1000]
Token 输入：     [1, 640, 100]
Backbone 输出：  [1, 640, 512]
分类特征：       [1, 327680]
类别概率：       [1, 4]
概率和：         约等于 1
重复推理：       输出一致
```

## 10. CLI Replay

先跑一个窗口：

```bash
python scripts/replay_offline.py \
  --config configs/stage05_50m_bnci_s01.yaml \
  --model-package runs/stage05_50m/model_package \
  --device cpu \
  --max-windows 1 \
  --replay-speed 100
```

再跑多个窗口：

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

## 11. Streamlit

```bash
streamlit run web/app.py
```

页面选择：

```text
Data:
data/processed/bnci2014_001/subject_01.h5

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

验证：

- 模型包能够发现并加载；
- 页面不再显示“未训练测试头”警告；
- Start 正常；
- Stop 正常；
- Restart 正常；
- EEG 波形更新；
- Prediction History 更新；
- Current / Average / P95 latency 更新；
- Maximum windows 达到后正常停止。

## 12. 预处理说明

当前暂定流程：

```text
单位统一为 µV
→ 映射到标准 64 通道
→ 缺失通道补零并生成 Mask
→ 可选平均参考，默认关闭
→ 原始采样率下 0.1–75 Hz 带通滤波
→ 重采样到 100 Hz
→ 严格检查真实 10 秒 / 1000 点
→ 有效通道按时间维 Z-score
```

目标采样率为 100 Hz，因此最终输入无法保留 50 Hz 以上的频率成分。0.1–75 Hz 仍是阶段 0.5 的临时设置，后续应继续与预训练 checkpoint 的真实数据管线核对。

## 13. 当前边界

1. 当前 Runtime 使用 10 秒输入配置。
2. 分类头是正式训练头，但训练窗口由同标签 4 秒 Trial 临时拼接得到。
3. Replay 中的普通 10 秒滑窗仍可能跨不同标签 Trial，因此单窗口 Replay 预测不适合作为严格准确率评估。
4. 模型文件和 HDF5 数据不会随 Git 仓库分发。
5. 当前未自动从火山云 TOS 下载 checkpoint。
6. 当前未完成 GPU 峰值显存统计。
7. 当前未接入真实 EEG 设备。
8. 当前未实现个体化、Rest-Tuning 和实时自适应。

## 14. 阶段 0.5 验收

- [ ] `python -m compileall src tests scripts web` 通过；
- [ ] `pytest` 通过；
- [ ] Backbone 无 missing keys；
- [ ] 正式分类头能够严格加载；
- [ ] Model Package 导出 Smoke Test 通过；
- [ ] Runtime Package 能够在新进程中重新加载；
- [ ] LaBraM Replay 回归通过；
- [ ] 50M CLI Replay 通过；
- [ ] 窗口数一致且 `failed_windows=0`；
- [ ] Streamlit Start / Stop / Restart 通过；
- [ ] 页面不显示测试头警告；
- [ ] Git 中不包含 checkpoint、HDF5、访问密钥和本机绝对路径。

## 15. 后续工作

1. 在 GPU 部署机记录 P95 延迟和峰值显存；
2. 将 10 秒窗口适配回 4 秒；
3. 重新训练 4 秒正式分类头；
4. 接入公司自采离线数据；
5. 接入 EEG 设备实时数据；
6. 增加简单个体化；
7. 升级 Rest-Tuning；
8. 最后加入在线自适应。

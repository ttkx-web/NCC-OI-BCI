# 50M Model Adapter

本目录提供公司 50M EEG 基座模型与 `NCC-OI-BCI` Pipeline 之间的模型适配层，用于阶段 0.5 的接口联调。

当前版本的目标是跑通：

```text
原始 10 秒 EEG 窗口
→ 50M 专用预处理
→ 通道映射与缺失通道 Mask
→ Patch / Token 构造
→ 冻结 50M Backbone
→ 线性任务分类头
→ prediction / probabilities / confidence
```

> 当前使用的 `test_linear_head.pt` 是未训练的测试分类头，仅用于验证完整推理链路。其预测类别和置信度没有准确率意义。

## 当前状态

已实现：

- 50M 部署配置；
- 标准 64 通道映射与通道别名归一化；
- 缺失通道补零及有效性 Mask；
- 0.1–75 Hz 带通滤波；
- 100 Hz 重采样；
- 10 秒严格窗口检查；
- 按通道 Z-score；
- `[64, 1000] → [640, 100]` Token 化；
- 50M Backbone 构建、冻结与部署 checkpoint 加载；
- 指定 Transformer 层的 Token Embedding 提取；
- Flatten / Mean 特征聚合；
- 线性任务分类头保存和加载；
- `predict_proba()`、`predict()` 和 `extract_embeddings()`；
- Pipeline 兼容预处理器；
- 统一 Runtime 构建入口；
- 随机输入 Smoke Test；
- BNCI 离线真实 10 秒窗口测试。

暂未实现：

- 正式运动想象线性分类头训练脚本；
- 50M Adapter 与 `SlidingWindowDecoder`、Replay 和 Streamlit 的正式接入；
- TOS 自动下载；
- GPU 显存峰值统计；
- 4 秒窗口适配；
- 可用于准确率评估的 10 秒单标签数据构建。

## 目录结构

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
├── smoke_test_50m_tokenization.py
├── smoke_test_50m_backbone.py
├── smoke_test_50m_classifier.py
├── smoke_test_50m_adapter.py
├── smoke_test_50m_runtime.py
└── test_50m_offline_window.py
```

## 输入契约

### Pipeline 原始窗口

```text
signal:       [C, T]
channel_names: 长度 C
sample_rate:  原始采样率
unit:         V / mV / uV / µV
```

阶段 0.5 要求输入是真实 10 秒窗口，不允许把 4 秒窗口补零到 10 秒。

例如 BNCI 数据原始采样率为 250 Hz 时：

```text
raw_window: [C, 2500]
```

### 50M 预处理输出

```text
signal:             [64, 1000] float32
channel_valid_mask: [64]       float32
```

### 50M Token 输入

```text
token_inputs:          [B, 640, 100] float32
token_channel_indices: [B, 640]      int64
token_time_indices:    [B, 640]      int64
token_valid_mask:      [B, 640]      float32
```

### Backbone 与分类输出

```text
token_embeddings: [B, 640, 512]
flatten_features: [B, 327680]
logits:           [B, num_classes]
probabilities:    [B, num_classes]
```

默认提取第 9 个 Transformer Block 后的表示，即 `output_layer_idx=8`。

## 环境安装

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate bci-dayloop
python -m pip install -e .
```

已经存在环境时：

```bash
conda activate bci-dayloop
python -m pip install -e .
```

## 外部文件

模型文件不提交到 Git。请在本地准备：

```text
checkpoints/model_deploy.pt
checkpoints/test_linear_head.pt
data/processed/bnci2014_001_s01.h5
```

### `model.pt`

必须是只包含 Tensor 和普通 Python 数据结构的部署 checkpoint。原始预训练 checkpoint 如果包含 `configs.Config` 对象，需要先在原 50M 仓库环境中转换成部署版本。

### `test_linear_head.pt`

这是由分类头 Smoke Test 生成的测试头，不是正式训练产物。

## 最小使用方式

```python
from bci_dayloop.models.model_50m.runtime import (
    build_50m_runtime_from_metadata,
)

runtime = build_50m_runtime_from_metadata(
    checkpoint_path="checkpoints/50m/model_deploy.pt",
    classifier_path="checkpoints/50m/test_linear_head.pt",
    metadata=metadata,
    device="cpu",
)

result = runtime.predict_raw_window(raw_window)

print(result.prediction)
print(result.confidence)
print(result.probabilities)
```

也可以将模型和预处理器分别交给 Pipeline：

```python
model = runtime.adapter
preprocessor = runtime.preprocessor

model_input = preprocessor.transform(raw_window)

probabilities = model.predict_proba(
    model_input[None, ...],
    channel_valid_masks=(
        preprocessor.last_channel_valid_mask[None, ...]
    ),
)[0]
```

## 测试顺序

建议按以下顺序运行，不要跳过中间层：

```bash
python scripts/smoke_test_50m_tokenization.py
python scripts/smoke_test_50m_backbone.py
python scripts/smoke_test_50m_classifier.py
python scripts/smoke_test_50m_adapter.py
python scripts/smoke_test_50m_runtime.py
```

真实 BNCI 10 秒窗口测试：

```bash
python scripts/test_50m_offline_window.py \
  --data data/processed/bnci2014_001_s01.h5 \
  --session 1test \
  --checkpoint checkpoints/50m/model_deploy.pt \
  --classifier checkpoints/50m/test_linear_head.pt \
  --device cpu \
  --window-sec 10 \
  --repeat 2 \
  --json-output runs/stage05_50m/offline_window_report.json
```

通过时应确认：

```text
预处理输出：     [64, 1000]
Token 输入：     [1, 640, 100]
Backbone 输出：  [1, 640, 512]
分类特征：       [1, 327680]
类别概率：       [1, 4]
概率和：         约等于 1
重复推理：       输出一致
```

## 交给 Replay 侧的唯一构建入口

Replay 接入方建议只使用：

```python
from bci_dayloop.models.model_50m.runtime import (
    build_50m_runtime_from_metadata,
)
```

```python
runtime = build_50m_runtime_from_metadata(
    checkpoint_path=checkpoint_path,
    classifier_path=classifier_path,
    metadata=metadata,
    device=device,
)

model = runtime.adapter
preprocessor = runtime.preprocessor
```

Replay 侧不应直接创建或修改：

- `Model50MBackbone`
- `Model50MClassifier`
- `Model50MTokenizer`
- `Model50MPreprocessor`

## 阶段 0.5 的 Replay 配置

阶段 0 的 LaBraM 配置继续保留 4 秒窗口。50M 使用独立配置：

```yaml
replay:
  window_sec: 10.0
  step_sec: 0.5
  speed: 1.0
  max_windows: 100
```

当前 BNCI HDF5 中每个 trial 只有 4 秒。10 秒 Replay 窗口由多个 trial 首尾拼接而成，可能同时包含多个类别，因此当前只能用于流程和延迟验证，不能用于准确率评估。

## 预处理说明

当前暂定流程：

```text
单位统一为 µV
→ 映射到标准 64 通道
→ 缺失通道补零并生成 Mask
→ 可选平均参考，默认关闭
→ 原采样率下 0.1–75 Hz 带通滤波
→ 重采样到 100 Hz
→ 固定为真实 10 秒 / 1000 点
→ 有效通道按时间维 Z-score
```

由于目标采样率是 100 Hz，最终输入无法保留 50 Hz 以上的频率成分。0.1–75 Hz 是当前临时设置，正式版本应与实际预训练 checkpoint 的数据管线再次确认。

## 当前限制

1. `test_linear_head.pt` 未训练，所有预测仅用于链路验证。
2. 模型文件和 HDF5 数据不会随 Git 仓库分发。
3. 当前 Runtime 只支持 10 秒原始配置。
4. 当前没有完成真实 Replay 和 Streamlit 接入。
5. 当前没有正式分类头训练脚本。
6. 当前没有自动从火山云 TOS 下载 checkpoint。
7. 当前没有在 `pytest` 中加入依赖真实 checkpoint 的集成测试。
8. 当前未完成 GPU 显存峰值记录。

## 交付验收

Adapter 接口交付前至少确认：

- [ ] 所有脚本不包含开发者本机绝对路径；
- [ ] `model_deploy.pt` 能以 `weights_only=True` 加载；
- [ ] Backbone 没有 missing keys；
- [ ] Tokenization Smoke Test 通过；
- [ ] Backbone Smoke Test 通过；
- [ ] Classifier 保存与重新加载测试通过；
- [ ] Adapter Smoke Test 通过；
- [ ] Runtime Smoke Test 通过；
- [ ] 真实 BNCI 10 秒窗口测试通过；
- [ ] 离线两种接口输出一致；
- [ ] README 中明确测试头不具备准确率意义；
- [ ] 分类头训练脚本的状态已明确标注。

## 后续工作

模型适配线：

1. 编写正式线性任务头训练脚本；
2. 建立可评估的单标签训练样本；
3. 记录模型加载时间、推理时间和显存峰值；
4. 将 10 秒输入改为 4 秒并重新训练任务头。

Pipeline 接入线：

1. 用 Runtime 构建入口替换 LaBraM 调用；
2. 显式传递 `channel_valid_mask`；
3. 接入 Replay、窗口统计和延迟统计；
4. 接入 Streamlit；
5. 完成 Start / Stop / Restart 和长时间运行测试。

# Stage 1：50M 简单个体化实验协议

> 文档状态：Draft  
> 适用阶段：Stage 1  
> 主线任务：BNCI2014_001 四分类运动想象  
> 当前方法：冻结 50M Backbone，仅更新线性分类头  
> 前置条件：Stage 0.5 已完成，50M Runtime、正式分类头、Model Package、CLI Replay 和 Streamlit 可稳定运行

---

## 1. 实验目的

Stage 1 用于验证：

> 对一个未参与群体模型训练的新被试，使用少量该被试的带标签 EEG 数据更新任务分类头后，个人模型是否比群体模型更适合该被试。

本阶段同时验证两类问题。

### 1.1 算法问题

1. 群体模型在完全未见目标被试上的性能如何；
2. 少量目标被试数据能否带来稳定个体化增益；
3. 个体化增益与数据量之间是什么关系；
4. 不同目标被试的收益是否一致；
5. 从群体分类头初始化是否优于随机初始化个人头。

### 1.2 工程问题

1. 个体化训练能否自动或半自动完成；
2. 个人分类头能否按用户保存、重新加载和隔离；
3. 群体模型与个人模型能否稳定切换；
4. 个人模型能否继续接入 CLI Replay 和 Streamlit；
5. 个人模型是否明显增加推理延迟；
6. 数据、配置、代码、模型和结果是否可追溯。

---

## 2. 本阶段边界

### 2.1 必须完成

- 使用多被试公开运动想象数据；
- 对每个目标被试执行 Leave-One-Subject-Out；
- 训练不包含目标被试的群体分类头；
- 从目标被试 `0train` 中抽取少量带标签 Trial；
- 冻结 50M Backbone，仅更新线性分类头；
- 在目标被试固定的 `1test` 上比较群体模型和个人模型；
- 测试每类 5、10、20、40 个 Trial；
- 使用多个随机种子重复实验；
- 输出 Accuracy、Macro-F1、Balanced Accuracy 和混淆矩阵；
- 保存个人模型、metadata、训练日志和评测结果；
- 验证个人模型保存、加载、切换和伪实时运行。

### 2.2 暂不包含

- Rest-Tuning；
- LoRA；
- Adapter 微调；
- 解冻 50M Backbone；
- 实时自适应；
- 公司自采数据上的正式算法结论；
- EEG 设备真实实时接入；
- 实体终端控制；
- 新任务或新范式。

本阶段只改变“分类头是否使用目标被试数据”，其余模型、预处理、标签和测试集保持一致。

---

## 3. 数据集与任务

### 3.1 数据集

默认使用：

```text
BNCI2014_001
```

默认被试：

```text
Subject 1–9
```

如部分被试文件缺失或数据准备失败，必须在实验报告中记录实际参与实验的被试列表，不能静默跳过。

### 3.2 任务

四分类运动想象：

```text
left_hand
right_hand
feet
tongue
```

类别顺序、数字标签和命令映射必须与 Stage 0.5 的 Runtime Model Package 保持一致。

### 3.3 Session

每个被试包含：

```text
0train
1test
```

二者用途在本协议中固定，不允许在不同实验中临时交换。

---

## 4. 目标被试划分

### 4.1 Leave-One-Subject-Out

对每个目标被试分别执行一次完整实验。

以 `Subject 1` 为例：

```text
目标被试：Subject 1
群体训练被试：Subject 2–9
```

对 `Subject 2`：

```text
目标被试：Subject 2
群体训练被试：Subject 1、3–9
```

依次遍历所有被试。

### 4.2 强制约束

对当前目标被试：

- 该被试的任何数据都不得用于群体分类头训练；
- 该被试的 `1test` 不得用于任何训练、超参数选择或 Early Stopping；
- 只有该被试的 `0train` 可以用于个体化；
- 群体模型和个人模型必须在完全相同的目标被试 `1test` 上比较。

---

## 5. 群体训练被试与 Session 用途

### 5.1 群体训练集

对于目标被试 `S_target`：

```text
Population Train:
所有非目标被试的 0train
```

### 5.2 群体验证集

```text
Population Validation:
所有非目标被试的 1test
```

群体验证集仅用于：

- 选择最佳 Epoch；
- Early Stopping；
- 选择分类头训练超参数；
- 检查群体模型是否正常收敛。

### 5.3 群体最终测试集

```text
Population Final Test:
目标被试的 1test
```

群体分类头训练完成后，直接在目标被试 `1test` 上评估，得到个体化前基线。

### 5.4 `0train` 与 `1test` 固定用途

| 数据来源 | 用途 | 是否参与参数更新 |
|---|---|---:|
| 非目标被试 `0train` | 群体分类头训练 | 是 |
| 非目标被试 `1test` | 群体分类头验证和 Early Stopping | 否 |
| 目标被试 `0train` 的个体化子集 | 个人分类头训练 | 是 |
| 目标被试 `0train` 的个人验证集 | 个人分类头验证和 Early Stopping | 否 |
| 目标被试 `1test` | 群体与个人模型最终独立测试 | 否 |

目标被试的 `1test` 在整个实验结束前必须保持封闭。

---

## 6. 个体化数据划分

### 6.1 抽样单位

个体化预算按“每类原始 Trial 数”定义：

```text
5 trials/class
10 trials/class
20 trials/class
40 trials/class
```

四分类对应总个体化 Trial 数：

| 每类 Trial 数 | 总 Trial 数 |
|---:|---:|
| 5 | 20 |
| 10 | 40 |
| 20 | 80 |
| 40 | 160 |

这里的 Trial 指目标被试 `0train` 中的原始带标签 Trial，不是 10 秒滑窗数量。

### 6.2 个人验证集

目标被试 `0train` 先划分为：

```text
Personalization Pool
Personal Validation
```

默认：

```text
每类固定保留 16 个 Trial 作为 Personal Validation
剩余 Trial 作为 Personalization Pool
```

如某个被试每类 Trial 数不足以同时支持 40 个训练 Trial 和 16 个验证 Trial：

1. 保证训练池与验证集不重叠；
2. 优先保证每类至少 8 个验证 Trial；
3. 在报告中记录实际验证 Trial 数；
4. 不得从 `1test` 中补充验证数据。

### 6.3 嵌套数据量

对同一个目标被试和同一个随机种子，5、10、20、40 Trial 子集必须满足：

```text
5 ⊂ 10 ⊂ 20 ⊂ 40
```

实现方式：

1. 对每个类别生成一次固定随机排列；
2. 前 5 个用于 5-trial 实验；
3. 前 10 个用于 10-trial 实验；
4. 前 20 个用于 20-trial 实验；
5. 前 40 个用于 40-trial 实验。

### 6.4 类别平衡

所有个体化实验必须保证：

```text
每个类别使用相同数量 Trial
```

若某类可用 Trial 不足，本次配置判定为不可运行，不允许通过重复采样补齐。

---

## 7. 10 秒窗口构造

### 7.1 保持 Stage 0.5 一致

Stage 1 继续使用 Stage 0.5 的 50M 输入协议：

```text
10 秒窗口
100 Hz
64 通道
严格窗口长度检查
```

### 7.2 原始 Trial 与 10 秒窗口

当前阶段继续沿用 Stage 0.5 的临时构造方案：

```text
同一被试
+ 同一 Session
+ 同一类别
的原始 Trial 按固定顺序拼接
→ 截取不重叠 10 秒窗口
```

### 7.3 强制约束

- 不得跨被试拼接；
- 不得跨 Session 拼接；
- 不得跨类别拼接；
- 不得使用目标被试 `1test` 构造训练窗口；
- 每个窗口必须记录来源 Trial ID；
- 训练集和验证集的来源 Trial ID 必须完全不重叠；
- 不允许用补零将短数据伪装为 10 秒；
- 默认使用非重叠窗口，`window_stride_sec=10`。

### 7.4 必须记录

```text
raw_trials_per_class
constructed_windows_per_class
dropped_samples
source_trial_ids
window_sec
window_stride_sec
```

因为“每类 5 个 Trial”不等于“每类 5 个 10 秒窗口”。

---

## 8. 模型设置

### 8.1 50M Backbone

整个 Stage 1 固定使用与 Stage 0.5 相同的：

- 50M 部署 checkpoint；
- 通道配置；
- 目标采样率；
- 滤波配置；
- 标准化配置；
- Tokenization；
- `output_layer_idx`；
- 特征聚合方式；
- 类别顺序。

默认 Backbone：

```text
checkpoints/backbones/50m/model_deploy.pt
```

### 8.2 Backbone 冻结

第一版所有实验：

```text
50M Backbone 全部冻结
```

必须验证：

```python
all(parameter.requires_grad is False for parameter in backbone.parameters())
```

训练日志必须记录：

```text
trainable_backbone_parameters = 0
```

### 8.3 可训练参数

第一版只允许更新：

```text
Linear Classifier Head
```

不得更新 Backbone、Tokenizer、LayerNorm、Channel Embedding、Positional Embedding 或预处理统计量。

---

## 9. 对照方法

### 9.1 方法 A：群体模型

```text
冻结 50M Backbone
+ 使用非目标被试训练群体分类头
```

用途：

- 个体化前基线；
- 个人分类头初始化；
- 群体与个人模型伪实时对照。

### 9.2 方法 B：简单个体化

```text
加载方法 A 的群体分类头
→ 冻结 50M Backbone
→ 使用目标被试少量 0train 数据继续更新分类头
```

这是 Stage 1 的正式个体化方案。

### 9.3 方法 B0：随机初始化个人头（建议对照）

```text
冻结 50M Backbone
+ 随机初始化分类头
+ 只使用目标被试少量数据训练
```

用途：判断群体头初始化是否有价值。

B0 是建议对照，不阻塞 Stage 1 最小闭环验收。

### 9.4 暂不执行

- 解冻最后若干 Transformer Block；
- LoRA；
- Adapter；
- 全量 Backbone 微调。

---

## 10. 分类头初始化与训练

### 10.1 群体分类头

群体分类头随机初始化，然后使用非目标被试数据训练。

### 10.2 个人分类头

正式方法 B 从目标被试对应的群体分类头初始化。

不能从其他目标被试的个人分类头初始化。

### 10.3 第一版训练配置

优先沿用 Stage 0.5 分类头训练配置：

```yaml
optimizer: sgd
learning_rate: 0.001
momentum: 0.0
weight_decay: 0.001
epochs: 100
patience: 15
metric_for_best: val_bacc
```

如 Stage 0.5 的实际正式配置不同，以正式配置为准，并同步修改本协议和配置文件。

### 10.4 Early Stopping

- 群体头：根据 Population Validation 的 Balanced Accuracy；
- 个人头：根据 Personal Validation 的 Balanced Accuracy；
- 不得根据目标被试 `1test` 选择最佳 Epoch。

### 10.5 特征缓存

由于 Backbone 冻结，优先采用：

```text
先提取并缓存 50M 特征
→ 再训练线性分类头
```

缓存至少包含：

```text
features
labels
subject_id
session_id
trial_ids
window_ids
backbone_hash
preprocessing_hash
feature_dtype
```

不同 Backbone 或预处理配置不得共用缓存。

---

## 11. 随机种子

### 11.1 开发阶段

最小闭环：

```text
42
```

批量调试：

```text
42, 43, 44
```

### 11.2 正式实验

```text
42, 43, 44, 45, 46
```

### 11.3 控制范围

每个 `run_seed` 必须控制：

- Python `random`；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- Trial 随机排列；
- DataLoader shuffle；
- 分类头初始化；
- Batch 顺序。

### 11.4 可复现性记录

每次运行保存：

```text
run_seed
Python version
PyTorch version
CUDA version
device
git commit
config hash
backbone hash
dataset file metadata
```

---

## 12. 评测指标

### 12.1 主要算法指标

必须报告：

```text
Accuracy
Macro-F1
Balanced Accuracy
```

### 12.2 补充指标

建议报告：

```text
每类 Precision
每类 Recall
每类 F1
混淆矩阵
预测置信度
```

### 12.3 个体化增益

```text
Accuracy Gain
= Personal Accuracy - Population Accuracy

Macro-F1 Gain
= Personal Macro-F1 - Population Macro-F1

Balanced Accuracy Gain
= Personal BAcc - Population BAcc
```

所有 Gain 使用绝对百分点表示。

例如：

```text
0.62 -> 0.68
Gain = +6.0 percentage points
```

### 12.4 数据量曲线

必须绘制：

```text
x 轴：5 / 10 / 20 / 40 trials per class
y 轴：Accuracy / Macro-F1 / Balanced Accuracy
```

至少包含：

- 每个目标被试曲线；
- 所有被试均值曲线；
- 随机种子标准差或 95% 置信区间。

---

## 13. 工程指标

每个个人模型必须记录：

```text
个体化训练总时间
最佳 Epoch
个人分类头文件大小
模型保存时间
模型加载时间
群体头 -> 个人头切换时间
个人头 -> 群体头切换时间
切换是否需要重启 Pipeline
切换过程中丢失窗口数
平均推理延迟
P95 推理延迟
窗口完成率
异常窗口数
```

由于 A 和 B 使用相同 Backbone，仅分类头参数不同，个人模型不应显著增加单窗口推理延迟。

---

## 14. 统计汇总

### 14.1 单次运行

每个组合输出一条完整结果：

```text
target_subject
trials_per_class
seed
```

### 14.2 被试内汇总

对每个目标被试和数据量，对正式 5 个 Seed 求均值和标准差。

### 14.3 跨被试汇总

对每个数据量报告：

```text
Population 指标均值
Personal 指标均值
平均 Gain
Gain 中位数
正增益被试数
非负增益被试比例
```

### 14.4 显著性分析

正式报告建议在被试级平均结果上执行配对比较：

```text
Personal vs Population
```

被试数量较少时，优先使用 Wilcoxon signed-rank test，并报告被试级 Gain 的 Bootstrap 95% 置信区间。

统计检验不能替代逐被试结果展示。

---

## 15. 实验输出目录

```text
runs/stage1/
├── feature_cache/
│   ├── subject_01_0train.pt
│   ├── subject_01_1test.pt
│   └── ...
│
├── target_subject_01/
│   ├── population/
│   │   ├── classifier.pt
│   │   ├── config.yaml
│   │   ├── training_report.json
│   │   └── test_metrics.json
│   │
│   ├── personal/
│   │   ├── trials_05_seed_42/
│   │   │   ├── classifier.pt
│   │   │   ├── metadata.json
│   │   │   ├── training_report.json
│   │   │   └── test_metrics.json
│   │   ├── trials_10_seed_42/
│   │   ├── trials_20_seed_42/
│   │   └── trials_40_seed_42/
│   │
│   └── comparison_report.json
│
├── target_subject_02/
│   └── ...
│
└── results/
    ├── personalization_results.csv
    ├── subject_summary.csv
    ├── overall_summary.json
    ├── data_scaling_accuracy.png
    ├── data_scaling_macro_f1.png
    └── confusion_matrices/
```

---

## 16. 模型 metadata

### 16.1 群体模型 metadata

至少包含：

```json
{
  "model_type": "50m_population_head",
  "target_subject": "subject_01",
  "excluded_subjects": ["subject_01"],
  "training_subjects": ["subject_02", "subject_03"],
  "population_train_sessions": ["0train"],
  "population_validation_sessions": ["1test"],
  "backbone_checkpoint": "checkpoints/backbones/50m/model_deploy.pt",
  "backbone_sha256": "...",
  "preprocessing_hash": "...",
  "classes": ["left_hand", "right_hand", "feet", "tongue"],
  "run_seed": 42,
  "git_commit": "..."
}
```

### 16.2 个人模型 metadata

至少包含：

```json
{
  "model_type": "50m_personal_head",
  "user_id": "subject_01",
  "base_population_model": "...",
  "task": "motor_imagery_4class",
  "personalization_session": "0train",
  "final_test_session": "1test",
  "trials_per_class": 20,
  "run_seed": 42,
  "source_trial_ids": {},
  "personal_validation_trial_ids": {},
  "training_time_sec": 0.0,
  "best_epoch": 0,
  "backbone_sha256": "...",
  "preprocessing_hash": "...",
  "git_commit": "..."
}
```

---

## 17. 统一结果表字段

```text
target_subject
trials_per_class
seed
population_accuracy
personal_accuracy
accuracy_gain
population_macro_f1
personal_macro_f1
macro_f1_gain
population_balanced_accuracy
personal_balanced_accuracy
balanced_accuracy_gain
population_loss
personal_loss
personal_best_epoch
personal_training_time_sec
personal_head_size_mb
personal_model_load_time_ms
model_switch_time_ms
population_mean_latency_ms
population_p95_latency_ms
personal_mean_latency_ms
personal_p95_latency_ms
window_completion_rate
failed_windows
git_commit
config_hash
```

---

## 18. 伪实时验证

### 18.1 输入

使用目标被试固定的：

```text
1test
```

### 18.2 对照方式

对同一个 EEG 窗口：

```text
共享一次 50M 特征提取
├── Population Head
└── Personal Head
```

群体和个人模型必须使用：

- 相同原始窗口；
- 相同预处理；
- 相同 Backbone 特征；
- 相同标签映射；
- 相同 Ground Truth。

### 18.3 逐窗口记录

```text
window_id
source_time
ground_truth
population_prediction
population_confidence
personal_prediction
personal_confidence
population_correct
personal_correct
feature_extraction_latency
population_head_latency
personal_head_latency
total_latency
```

### 18.4 指标准确性说明

普通 Replay 的 10 秒窗口可能跨越原始 Trial 标签，因此：

- 严格 Accuracy、Macro-F1 和混淆矩阵使用标签明确的独立评测窗口；
- 普通连续 Replay 主要验证模型加载、切换、稳定性、延迟和展示；
- 不得把跨标签 Replay 窗口的预测直接作为正式算法准确率。

---

## 19. 数据泄漏检查

每次实验启动前必须自动检查。

### 19.1 被试泄漏

```text
target_subject not in population_training_subjects
target_subject not in population_validation_subjects
```

### 19.2 Session 泄漏

```text
target 1test 不参与任何训练和验证
```

### 19.3 Trial 泄漏

```text
personal_train_trial_ids
∩ personal_validation_trial_ids
= empty

personal_train_trial_ids
∩ final_test_trial_ids
= empty

personal_validation_trial_ids
∩ final_test_trial_ids
= empty
```

### 19.4 窗口泄漏

由同一个原始 Trial 生成的窗口不得同时出现在训练集和验证集。

必须先在 Trial 层面划分，再构造 10 秒窗口。

---

## 20. 验收标准

### 20.1 功能验收

- [ ] 多被试数据能够读取；
- [ ] LOSO 划分正确；
- [ ] 目标被试未进入群体训练；
- [ ] 目标被试 `1test` 未进入训练和验证；
- [ ] 5/10/20/40 Trial 抽样可复现；
- [ ] 同一 Seed 的数据量子集满足嵌套关系；
- [ ] Backbone 全部冻结；
- [ ] 只有线性分类头参数发生更新；
- [ ] 群体模型可以保存和重新加载；
- [ ] 个人模型可以保存和重新加载；
- [ ] 重新加载后输出一致；
- [ ] 不同用户模型能够正确隔离；
- [ ] 群体模型和个人模型能够切换；
- [ ] 切换后 Pipeline 可继续运行；
- [ ] 日志完整；
- [ ] 结果可追溯到数据、配置、代码和模型版本。

### 20.2 算法验收

Stage 1 不要求每个被试都提升，阶段目标为：

```text
多数目标被试个体化后不下降
且跨被试平均 Accuracy / Macro-F1 / Balanced Accuracy 获得正增益
```

正式报告必须展示：

- 所有被试结果；
- 下降或无增益被试；
- 不同数据量结果；
- Seed 方差；
- 平均增益和中位数增益；
- 个体化无效或负增益案例分析。

不得只报告表现最好的被试或随机种子。

### 20.3 工程验收

- [ ] 个体化训练可以正常启动和结束；
- [ ] 个人分类头可按用户保存；
- [ ] 个人分类头可在新进程中加载；
- [ ] 模型加载和切换时间可记录；
- [ ] 切换不需要重启整个 Pipeline，或明确记录当前限制；
- [ ] 切换过程中无窗口丢失，或能够记录暂停时间；
- [ ] 个人模型平均/P95 延迟与群体模型无异常差异；
- [ ] 伪实时窗口完成率满足 Stage 0.5 基线；
- [ ] 无新增异常窗口和程序崩溃。

---

## 21. 实施顺序

### Milestone 1：实验协议与数据结构

完成：

- 本实验协议；
- 多被试 HDF5 数据结构；
- Trial ID、Subject ID、Session ID 保存；
- LOSO 划分接口；
- 数据泄漏单元测试。

### Milestone 2：单目标被试群体模型

固定：

```text
target_subject = Subject 1
```

完成：

```text
Subject 2–9 的 0train
→ 训练群体分类头
→ Subject 2–9 的 1test 验证
→ Subject 1 的 1test 最终测试
```

### Milestone 3：单目标被试个体化

固定：

```text
target_subject = Subject 1
trials_per_class = 20
seed = 42
```

完成：

```text
加载群体头
→ Subject 1 的 0train 少量数据个体化
→ Subject 1 的 1test 测试
→ 保存和重新加载个人头
```

### Milestone 4：数据量实验

增加：

```text
5 / 10 / 20 / 40 trials per class
```

先使用：

```text
seed = 42
```

### Milestone 5：多随机种子

先增加：

```text
42 / 43 / 44
```

流程稳定后增加到：

```text
42 / 43 / 44 / 45 / 46
```

### Milestone 6：所有目标被试

依次执行 Subject 1–9 LOSO。

### Milestone 7：模型管理和伪实时

完成：

- 用户模型目录；
- 模型 metadata；
- 群体/个人模型切换；
- 双轨预测日志；
- CLI 或 Streamlit 对比展示。

### Milestone 8：阶段报告

输出：

- 完整 CSV；
- 被试级结果表；
- 数据量—性能曲线；
- 混淆矩阵；
- 个体化前后性能卡片；
- 工程指标；
- 已知问题和失败案例。

---

## 22. 第一轮最小实验

第一轮只执行：

```yaml
dataset: BNCI2014_001
target_subject: 1
population_train_subjects: [2, 3, 4, 5, 6, 7, 8, 9]
population_train_session: 0train
population_validation_session: 1test
personalization_session: 0train
final_test_session: 1test

trials_per_class: 20
personal_validation_trials_per_class: 16
seed: 42

freeze_backbone: true
train_classifier_only: true
personal_head_init: population_head

window_sec: 10
window_stride_sec: 10
target_sample_rate: 100
```

最小闭环完成条件：

```text
群体模型训练成功
→ 在 Subject 1 / 1test 上得到基线
→ 个人模型训练成功
→ 在相同 Subject 1 / 1test 上完成比较
→ 个人模型成功保存
→ 新进程重新加载结果一致
→ 输出 comparison_report.json
```

在该闭环完成前，不开始 LoRA、Streamlit 页面扩展或所有被试批量实验。

---

## 23. 建议新增文件

```text
configs/
└── stage1_50m_bnci_loso.yaml

scripts/
├── prepare_bnci2014_001_multisubject.py
├── train_50m_population_head.py
├── train_50m_personal_head.py
├── evaluate_stage1_personalization.py
├── run_stage1_personalization_benchmark.py
└── replay_compare_population_personal.py

src/bci_dayloop/personalization/
├── __init__.py
├── split.py
├── trainer.py
├── registry.py
├── package.py
└── metrics.py

tests/
├── test_personalization_split.py
├── test_personalization_sampling.py
├── test_personalization_no_leakage.py
├── test_personal_head_save_load.py
├── test_user_model_registry.py
└── test_population_personal_replay.py
```

---

## 24. 推荐提交顺序

```text
1. docs: add stage 1 personalization protocol
2. feat: add multi-subject BNCI data preparation
3. feat: add leave-one-subject-out split utilities
4. test: add stage 1 leakage checks
5. feat: add 50M population head training
6. feat: add few-shot personal head training
7. feat: add personalization evaluation benchmark
8. feat: add user model registry
9. feat: add population and personal replay comparison
10. docs: add stage 1 results and validation report
```

---

## 25. 协议变更规则

正式批量实验开始后，以下内容不得无记录修改：

- 目标被试定义；
- `0train` / `1test` 用途；
- Trial 数据量；
- 随机种子；
- Backbone checkpoint；
- 预处理参数；
- 特征聚合方式；
- 分类头结构；
- Early Stopping 指标；
- 最终测试集；
- 主要评测指标。

如确需修改：

1. 更新本协议版本；
2. 更新配置文件；
3. 记录修改原因；
4. 重新运行所有受影响实验；
5. 不得将修改前后结果混在同一个总表中。

---

## 26. 最终交付物

Stage 1 完成时应交付：

```text
1. 多被试数据准备脚本
2. LOSO 数据划分模块
3. 群体分类头训练脚本
4. 简单个体化训练脚本
5. 批量实验脚本
6. 用户模型保存、加载和隔离模块
7. 群体/个人模型伪实时对比脚本
8. 单元测试与数据泄漏检查
9. 全部被试、数据量和随机种子的结果表
10. Accuracy / Macro-F1 / BAcc 数据量曲线
11. 被试级混淆矩阵
12. 个体化训练时间和模型管理指标
13. Stage 1 验证报告
14. 可供 Stage 3 复用的个人模型包
```

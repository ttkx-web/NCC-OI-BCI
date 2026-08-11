# CBRaMod 冻结骨干基线实验协议

## 1. 目的与适用范围

本协议定义 CBRaMod 在 NCC-OI-BCI 中的**群体预训练基线**。它回答的问题是：在相同的下游任务、被试隔离方式和测试集下，冻结 CBRaMod 预训练编码器后，仅训练任务分类头，能够取得怎样的跨被试运动想象解码性能和端到端推理开销。

本协议的首个任务为 `BNCI2014_001` 四分类运动想象（`left_hand`、`right_hand`、`feet`、`tongue`），使用 4 秒 direct-trial 输入。首个目标被试为 `subject_01`，后续可按相同规则扩展到所有目标被试。

本协议不是：

- CBRaMod 论文中全量微调结果的复现；
- Rest-Tuning、NeuroOnline 或任意在线更新实验；
- 使用目标被试数据训练群体模型的实验；
- 以连续 Replay 重叠窗口准确率替代有标签 trial 级测试的实验。

## 2. 固定基线定义

实验名称固定为 `cbramod-frozen-head`。

| 项目 | 固定定义 |
| --- | --- |
| 预训练骨干 | 官方发布的 CBRaMod 网络定义与预训练 checkpoint；在一次实验中二者版本必须固定并记录 SHA-256。 |
| 可训练参数 | **仅分类头**。CBraMod 的 patch embedding、criss-cross transformer、归一化层和预训练输出投影均不更新。 |
| 骨干模式 | 训练和评估时均使用 `eval()`；不更新 BatchNorm 统计量，不启用骨干 dropout。 |
| 分类头 | 首版采用官方 quick-start 的 MLP 模板：`Flatten → Linear(22×4×200, 800) → ELU → Dropout(0.1) → Linear(800, 200) → ELU → Dropout(0.1) → Linear(200, 4)`。 |
| 损失 | 四分类交叉熵；类别权重默认关闭。若启用，必须对所有对比模型使用同一由训练集计算的规则。 |
| 最优 checkpoint | 以群体验证集 balanced accuracy 最大为准；禁止使用目标被试 `1test` 选择 epoch 或超参数。 |
| 随机性 | 固定并报告 `seed`；每个正式配置至少运行 3 个种子。 |

该基线是“冻结骨干 + MLP 分类头训练”，**不是 linear probe**。结果表、图和文件名不得将它称为 `linear`，以免将分类头容量差异错误归因于预训练骨干。

如需补充严格线性探针，应另建实验名 `cbramod-frozen-linear`，并单独报告；不得与本协议的主结果混合或取较好者。

## 3. 数据、划分与泄漏控制

### 3.1 数据与标签

- 数据集：`BNCI2014_001`。
- 原始 trial：使用与当前 Stage 1 相同的 `trial_tmin_sec=2.0`、`trial_tmax_sec=6.0`，即每条 trial 的 4 秒运动想象片段。
- 标签空间与顺序固定为：`[left_hand, right_hand, feet, tongue]`，并写入分类头 checkpoint、model package 和结果文件。
- 每一条输入必须保留 `subject_id`、`session`、`trial_id`、`label` 与原始时间范围；这些元数据不得在特征缓存后丢失。

### 3.2 群体模型的 LOSO 划分

对于目标被试 `S_target`：

| 数据来源 | 用途 | 是否允许更新参数 |
| --- | --- | ---: |
| 所有非目标被试的 `0train` | 群体分类头训练 | 是，仅分类头 |
| 所有非目标被试的 `1test` | 群体验证、early stopping、超参选择 | 否 |
| 目标被试的 `0train` | 本协议中完全不用；预留给后续个体化 | 否 |
| 目标被试的 `1test` | 一次性最终独立测试 | 否 |

因此，`subject_01` 的首个实验为：Subject 2--9 的 `0train` 训练、Subject 2--9 的 `1test` 验证、Subject 1 的 `1test` 最终测试。所有目标被试都必须分别训练一个不含该被试的群体头；不得将某一个目标被试的结果称为总体平均性能。

### 3.3 明确禁止的泄漏

- 目标被试的任意 `0train` 或 `1test` 不得参与群体头训练、标准化统计、类别权重计算、early stopping 或超参数选择。
- 目标被试 `1test` 只能在超参数完全冻结后运行一次正式评估。
- 同一原始 trial 派生的样本不得跨 train/validation/test。
- 不得为了补足输入长度跨 trial、跨类别或跨 session 拼接信号。

## 4. CBRaMod 输入与预处理契约

CBraMod 不接受当前 50M 的 `[64, T]` token 化输入。其官方 quick-start 所示的 BCI-IV-2a 输入为：

```text
[batch, channels, time_segments, points_per_patch]
= [B, 22, 4, 200]
```

本项目的 `CBraModPipelinePreprocessor` 必须将一个有标签 4 秒 trial 转成上述格式，并在训练、离线评估、Runtime Package 与 Replay 中由同一实现调用。

下列项目在第一次正式训练前必须写入不可变的 `preprocessing.json`；未完成核验不得产出正式对比结果：

| 字段 | 要求 |
| --- | --- |
| `source_channel_names` | HDF5 内真实通道名与顺序。 |
| `target_channel_names` | 与所用官方 checkpoint 对应的精确 22 通道顺序；不得只检查通道数为 22。 |
| `channel_mapping` | 从源通道到目标通道的一一映射；缺失、重复或未知通道必须报错。 |
| `source_sample_rate_hz` / `target_sample_rate_hz` | 分别记录实际输入采样率和 checkpoint 要求的采样率。只有在 `4 × target_sample_rate_hz = 800` 时，才能得到 `[22, 4, 200]`。 |
| `filter` | 是否滤波、频段、阶数和零相位设置；不得沿用 50M 的设置而不记录。 |
| `normalization` | 统计范围、公式、eps 及统计量的拟合来源。训练集统计量只能由非目标被试 `0train` 得出。 |
| `patching` | 固定为四个相邻、无重叠的 1 秒 patch；不得对单条 4 秒 trial 补零或截断。 |
| `implementation_hash` | 预处理代码版本和配置内容的哈希。 |

当前 `[22, 4, 200]` 是以官方公开示例为依据的**目标张量形状**，不等同于已经确认的官方 checkpoint 预处理规范。通道顺序、采样率、滤波与归一化须通过官方数据集封装、checkpoint 前向 smoke test 和可复现样例共同核验后才冻结。

## 5. 训练流程

1. 加载并校验官方 CBRaMod checkpoint；记录仓库 commit、权重下载来源、文件 SHA-256 与模型参数数量。
2. 构建 CBRaMod 骨干，替换/旁路预训练任务输出投影，使其输出与官方 quick-start 分类头输入一致。
3. 将所有骨干参数 `requires_grad=False`，优化器参数列表只能来自分类头；训练启动时断言骨干可训练参数数目为 0。
4. 用群体训练集拟合允许拟合的预处理统计量，并把统计量固化到本次运行目录；验证和最终测试仅调用 `transform`。
5. 训练分类头，验证集 bACC 改善时保存 best checkpoint；记录每 epoch 的 loss、accuracy、macro-F1、bACC、学习率和耗时。
6. 训练结束后载入 best checkpoint，在未触及的目标被试 `1test` 上只执行一次最终评估。

超参数搜索只允许使用非目标被试验证集。第一版应固定分类头结构、optimizer 类型和候选超参数网格，再统一用于 LaBraM、50M 和 CBRaMod 各自的训练流程；若模型需要不同学习率范围，应在配置中显式声明并完整报告搜索空间。

## 6. 指标与报告

### 6.1 任务性能

在目标被试 `1test` 的**trial 级**预测上报告：

- Accuracy；
- Balanced Accuracy（主指标）；
- Macro-F1；
- 混淆矩阵；
- 每个种子的结果，以及均值和标准差。

所有指标均使用 4 个类别的 argmax 预测。没有类别概率或标签时，不得将 Replay 产生的连续窗口加入上述指标。

### 6.2 运行时性能

将导出的 `cbramod-frozen-head` Runtime Model Package 接入相同的 4 秒窗口 Replay，单独报告：模型加载时间、每窗口预处理时间、模型前向时间、端到端平均延迟、P50、P95、峰值内存和窗口完成率。

Replay 的 `window_sec=4.0` 和 `step_sec=0.5` 用于观察持续输出与端到端延迟；它不是对 trial 级 bACC 的替代评测。若需要 Replay 准确率，必须另行定义窗口标签对齐规则、边界丢弃规则和统计单位。

### 6.3 结果记录

每次正式运行必须保存：

```text
runs/stage1/cbramod/
  target_subject_XX/
    frozen_head/
      config.yaml
      preprocessing.json
      checkpoint_manifest.json
      training_history.jsonl
      best_head.pt
      final_metrics.json
      confusion_matrix.csv
      runtime_metrics.json
```

`final_metrics.json` 至少包括模型名、目标被试、所有训练/验证被试、split、seed、best epoch、四项任务指标、权重/预处理哈希、推理设备与时间戳。

## 7. 与现有基线的对比规则

CBraMod、LaBraM 和 50M 的对比必须满足：

1. 相同数据版本、目标被试、标签定义、trial 时间范围和 LOSO 划分；
2. 每个模型使用其已记录的模型专属预处理，不能为了表面一致性强行采用其他模型的通道/采样率契约；
3. 不同分类头容量必须在结果表中如实标示；不能把 `cbramod-frozen-head` 与“线性头”混称为同一种 probe；
4. 离线任务性能与 Runtime/Replay 延迟分成两张表报告；
5. 主结论基于每个目标被试独立得到的 LOSO 结果，而非只报告 Subject 1。

## 8. 验收标准

开始正式实验前，以下项目必须全部通过：

- [ ] 官方 checkpoint 能在 CPU 上加载，且单 batch 前向成功；
- [ ] 预处理后的 shape 严格为 `[B, 22, 4, 200]`，并通过通道名与采样率断言；
- [ ] 训练中骨干可训练参数数为 0，骨干权重哈希在训练前后不变；
- [ ] train/validation/test 的 `trial_id` 交集为空；
- [ ] 最优 epoch 只由非目标被试 `1test` 的 bACC 决定；
- [ ] best head 能导出并由 Runtime Package 独立加载；
- [ ] 直接离线推理与 package 推理在同一 trial 上 logits/prediction 一致；
- [ ] 最终结果包含任务指标、运行时指标、权重哈希和预处理哈希。

## 9. 后续扩展边界

在本协议稳定后，才可新增以下独立实验：`cbramod-frozen-linear`、`cbramod-full-finetune`、基于 `cbramod-frozen-head` 的个体化分类头、Rest-Tuning 和 NeuroOnline。任何扩展均不得覆盖或改写本协议产生的冻结骨干基线结果。

# Awakening 2 秒 frozen-50M 训练准备

本链路读取 `data/processed/awakening_2s/subject_*.h5`，复用 flat-root
`EEGHDF5` / `LegacyHDF5Adapter`、现有 50M 预处理器、tokenizer、backbone、
linear head、early stopping 和公共分类指标。它不会改写 H5。

## 数据与 split 契约

- 数据集名：`awakening`；canonical subject 为 `1..21`。
- 类别顺序固定为 `[non_awakening, awakening]`，AUROC 正类索引固定为 `1`。
- `S1` 是 population train，`S2` 是 population validation / target final test。
- `S1/S2` 是转换阶段按 `source_lance_row_id` 确定性生成的逻辑划分，**不是原始实验 session**。
- 同一 Lance 父行的子窗不能跨 S1/S2。每次缓存前都会扫描 trial-level provenance
  并拒绝父行跨 split 或跨 subject。
- LOSO target subject 不进入 population train/validation。以 target 1 为例：被试
  2–21 的 S1/S2 分别训练和选模，被试 1 的 S2 只用于最终离线评估。
- `sub-1037`（canonical subject 8）只有类别 1；校验只要求整体 train 和
  validation 各自包含两个类别，不要求每个被试都包含两个类别。

该逻辑 split 适合快速训练和联调，不等价于严格跨被试性能结论。对外报告泛化性能
时，应逐 target subject 执行 LOSO，或使用按 `subject_id` 分组的 GroupKFold；禁止
窗口级随机划分。

## 50M 预处理契约

配置源是 `configs/datasets/awakening_2s.yaml`，Python 注册项是
`bci_dayloop.data.dataset_registry.AWAKENING_2S`。

```text
H5/raw                 [B,62,400] @ 200 Hz, uV
空间通道映射后         [B,64,400]
抗混叠 resample_poly   [B,64,200] @ 100 Hz
tokens                 [B,128,100]
flatten feature        [B,65536]
linear logits          [B,2]
```

其中 60 个源通道映射到 50M 标准通道；`TP9/TP10` 被忽略；缺失的
`AF7/AF8/F9/F10` 使用既有空间补零和 channel-valid mask。这不是时间补零。
200→100 Hz 仅由公共预处理器执行一次，输入已经是 2 秒，因而时间补零和裁剪均为 0。
输入单位已经是 `uV`，不会额外放大。`docs/standard_64_channels.json` 与实际 50M
代码都以 `F9/F10` 结尾。

## 1. 生成完整 frozen feature cache

先执行只读预检：

```bash
python scripts/cache_50m_awakening_features.py \
  --data-root data/processed/awakening_2s \
  --backbone-checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --target-subject 1 \
  --dry-run
```

正式生成（本次代码交付未执行）：

```bash
python scripts/cache_50m_awakening_features.py \
  --data-root data/processed/awakening_2s \
  --cache-dir data/features/awakening_2s/50m_frozen_target01 \
  --backbone-checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --target-subject 1 \
  --device cuda \
  --feature-batch-size 8 \
  --feature-cache-dtype float16 \
  --seed 42
```

缓存按 split × subject 分片，避免一次载入全部 EEG；完成后才原子替换目标目录。
默认拒绝已有目标，只有显式 `--overwrite` 才替换。写入前检查磁盘余量。

`cache_manifest.json` 和每个 shard 的强校验 contract 绑定：数据三份 manifest 的
SHA256、subjects/target subject、S1/S2 协议、窗口顺序 seed、缓存 dtype、窗口长度、
源/目标采样率、完整 channel profile、预处理版本和 hash、backbone 路径和 SHA256、
output layer、aggregation、feature dim、class mapping 和正类索引。训练或评估发现任意字段不一致会 fail-fast，
因此不能误用 10 秒、其他任务、其他通道模板或其他 backbone 的缓存。

## 2. 训练 frozen binary linear head

```bash
python scripts/train_50m_awakening_head.py \
  --data-root data/processed/awakening_2s \
  --cache-dir data/features/awakening_2s/50m_frozen_target01 \
  --backbone-checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --target-subject 1 \
  --device cuda \
  --epochs 100 \
  --head-batch-size 32 \
  --head-lr 0.001 \
  --weight-decay 0.001 \
  --patience 15 \
  --class-weight balanced \
  --cache-seed 42 \
  --feature-cache-dtype float16 \
  --seed 42 \
  --output checkpoints/heads/stage1/awakening/subject_01/population/2s_flatten/head.pt \
  --run-dir runs/stage1/awakening/subject_01/population/2s_flatten/run_001
```

只有 linear head 参数进入 AdamW；backbone 始终冻结。默认只使用 balanced
cross-entropy class weights，没有同时启用 balanced sampler，避免重复补偿。checkpoint
记录 task/class order/positive index、完整预处理和通道契约、split 语义、cache 和
backbone identity、seed 与训练参数。

可先使用 `--dry-run` 校验完整 data/cache contract 和两个 split 的类别覆盖；它不会
构造 optimizer、训练或写产物。

## 3. 对指定 checkpoint 离线评估

```bash
python scripts/evaluate_50m_awakening_head.py \
  --data-root data/processed/awakening_2s \
  --cache-dir data/features/awakening_2s/50m_frozen_target01 \
  --backbone-checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --classifier-checkpoint checkpoints/heads/stage1/awakening/subject_01/population/2s_flatten/head.pt \
  --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --target-subject 1 \
  --split target_final_test \
  --device cuda \
  --batch-size 32 \
  --cache-seed 42 \
  --feature-cache-dtype float16 \
  --output runs/evaluation/awakening_50m_subject01_s2.json
```

输出 loss、accuracy、balanced accuracy、macro-F1、AUROC、confusion matrix、
两类样本数和可计算的 per-subject 指标。AUROC 明确使用
`probabilities[:, 1]`；单类别被试的 per-subject AUROC 记为 `null`，其他指标照常给出。
评估 JSON 同时记录 checkpoint、split、class order、channel profile、预处理与 cache
contract hash。

## 轻量 smoke test

```bash
python scripts/smoke_test_50m_awakening.py \
  --data-root data/processed/awakening_2s \
  --subject 1 \
  --session S1 \
  --batch-size 1 \
  --backbone-checkpoint checkpoints/backbones/50m/model_deploy.pt \
  --device cpu
```

它只读取一个 H5 窗口并做一次 `torch.no_grad()` backbone/head forward，不训练、
不写 cache/checkpoint/run。CLI 完整参数以各脚本的 `--help` 为准。

## 范围边界

本入口只准备研究训练链路。它不导出 Runtime Model Package，不修改
`serve_inference.py`、Rust GUI、实时协议或任何 H5。接入 Runtime Package 前仍需独立
确认：目标 checkpoint 的审批与版本、正式 LOSO 汇总规则、阈值/校准策略、外部展示
文案；`non_awakening/awakening` 仍不可擅自解释成 `sedation/awake`。

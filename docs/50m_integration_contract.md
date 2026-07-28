# 50M 集成契约（待 A 确认）

本文件只定义待确认接口，不表示已接入 50M。

## 已确认的阶段 0.5 50M 基线

| 项目 | 已确认值 |
| --- | --- |
| 窗口与步长 | 10 秒窗口，0.5 秒步长 |
| 目标采样率 | 100 Hz |
| 输入形状 | `64 × 1000` |
| patch | 100 samples；10 个时间 patch；640 tokens |
| Backbone | embedding 维度 512；Transformer 深度 12；`output_layer_idx=8`；`aggregation=flatten` |
| 类别顺序 | `left_hand`、`right_hand`、`feet`、`tongue` |
| checkpoint | `E:\code\BCI_DayLoop\checkpoints\50M\pretrain_checkpoint_4.pt` |
| checkpoint SHA-256 | `97335B696B3AE9138DCB51C736F49EE1C6008FDC22FC42F13EA9A5301452F36E` |
| checkpoint 权重 | 顶层 key：`model_state_dict`；`time_embed.weight=(10,512)`；`channel_embed.weight=(64,512)`；`tokenizer.proj.0.weight=(512,100)` |
| objective | 基于权重与输出头结构，高置信度为 `timefreq` |

| 待确认项 | 当前约定 / 状态 |
| --- | --- |
| checkpoint | 待 A 确认 |
| 模型版本 | 待 A 确认 |
| 输入单位 | 待 A 确认 |
| 采样率 | 待 A 确认 |
| 64 通道顺序 | 待 A 确认 |
| 缺失通道规则 | 待 A 确认 |
| 输入/token shape | 待 A 确认 |
| patch 参数 | 待 A 确认 |
| embedding 层 | 待 A 确认 |
| pooling | 待 A 确认 |
| embedding 维度 | 待 A 确认 |
| 分类头 | 待 A 确认 |
| 类别顺序 | 待 A 确认 |
| 参考输入及参考 logits | 待 A 确认 |
| 保存加载格式 | 待 A 确认 |

## 10 秒数据合同待确认

- 当前 BNCI HDF5 中每个 trial 只有 4 秒。
- 10 秒训练窗口如何生成尚未最终确定。
- 窗口标签对应哪个时刻尚未确定。
- 跨 trial 或任务切换窗口如何处理尚未确定。
- 训练和伪实时 Replay 必须使用相同窗口规则。

50M Adapter、四分类分类头和可保存加载的模型包尚未交付；本合同不将其视为已完成接入。

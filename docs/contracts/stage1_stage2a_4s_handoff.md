# Stage 1 → Stage 2A: 4 s Offline Handoff

本合同固定 B 侧当前历史采集数据的离线输出；本轮不调用 A 侧模型代码。

## B 侧输出

- `signal`: `[59, 1000]`，保留原始 EEG 通道名与顺序；
- 采样率：250 Hz；
- 单位：`uV`；
- 时长：4.0 秒；
- `window_semantics="cue_plus_imagery_4s"`；
- `eligible_for_accuracy=true`；
- `accuracy_scope="cue_plus_imagery_task_classification"`；
- `visual_cue_present=true`，`visual_cue_duration_seconds=0.8`；
- `eligible_for_pure_imagery_accuracy=false`；
- 57 个通道可映射到标准 64 通道模板。

该窗口由历史范式的约 0.8 秒视觉类别提示与约 3.2 秒运动想象构成，不能命名为 `pure_imagery_4s` 或 `imagery_4s`。可以报告 Accuracy、Macro-F1、Balanced Accuracy、Confusion Matrix，以及 population/personal 对比，但必须使用以下披露：

> 当前分类指标反映包含视觉提示阶段和运动想象阶段的 cue-plus-imagery 四分类任务性能。模型可能利用视觉提示相关脑电活动，因此该结果不能直接等同于纯运动想象阶段解码性能。

每个 trial 必须随附标签、block/trial ID、BDF 权威的 `start_sample`/`end_sample`、BDF/CSV SHA-256、Reader 与单位证据 provenance。不得补零、重采样、预处理或改变通道名。

## A 侧预期（仅合同）

- `signal`: `[64, 400]`；
- `channel_valid_mask`: `[64]`，有效通道数 57；
- tokens: `[256, 100]`；
- feature shape: `[1, 131072]`。

未来重新采集的标准数据使用 `window_semantics="imagery_4s"`、`visual_cue_present=false`、`eligible_for_accuracy=true` 与 `eligible_for_pure_imagery_accuracy=true`；该语义与当前历史数据严格区分。

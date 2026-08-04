# Stage 1 → Stage 2A: 4 s Offline Handoff

本合同固定 B 侧当前历史采集数据的离线输出，不调用 A 侧模型代码。

## B 侧输出

- `signal`：`[59, 1000]`，原始 EEG 通道顺序；
- 采样率：250 Hz；
- 单位：`uV`；
- 时长：4.0 秒；
- `window_semantics="cue_plus_imagery_4s"`；
- `eligible_for_accuracy=false`；
- 57 个通道可映射到标准 64 通道模板。

该窗口由历史范式的 0.8 秒 cue 与 3.2 秒 imagery 构成，不能命名为 `pure_imagery` 或 `imagery_4s`，也不能据此输出正式分类准确率。

每个 trial 必须随附标签、block/trial ID、BDF 权威的 `start_sample`/`end_sample`、BDF/CSV SHA-256、Reader 与 unit evidence provenance。不得补零、重采样、预处理或改变通道名。

## A 侧预期（仅合同）

- `signal`：`[64, 400]`；
- `channel_valid_mask`：`[64]`，有效通道数 57；
- tokens：`[256, 100]`；
- feature shape：`[1, 131072]`。

未来重新采集的标准数据应使用 `window_semantics="imagery_4s"` 且 `eligible_for_accuracy=true`；该语义与当前历史数据严格区分。

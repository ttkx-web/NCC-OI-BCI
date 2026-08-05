# Neuracle Marker Mapping

Neuracle annotation description 会先解析整数 Marker code。支持整数、整数文本、`S 1`、`S1` 与 `Stimulus/S 1` 等形式。已定义的语义如下：

| Marker | event_type | label | 备注 |
| --- | --- | --- | --- |
| 1 | imagery | left_hand | |
| 2 | imagery | right_hand | |
| 3 | imagery | feet | 原始标签保留为 `both_feet` |
| 4 | imagery | tongue | |
| 10 | fixation | | |
| 20 | rest | | |
| 90 | block_start | | |
| 91 | block_end | | |
| 100 | recording_start | | |
| 101 | recording_end | | |
| 127 | abort | | |

未知整数 Marker 和无法解析为整数的 description 都保留为 `custom`，不会被丢弃。每个事件 metadata 保留 `original_description` 与 `marker_code`；Marker 3 还保留 `original_label=both_feet`。

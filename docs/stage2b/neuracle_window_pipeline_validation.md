# Neuracle JellyFish 实时窗口流水线验证（脱敏）

## 范围

本记录验证以下实时链路的窗口行为：

`NeuracleJellyFishSource → EEG-only selector → TimestampBuffer → 4 s windowing`

它只记录脱敏的运行统计，不保存 EEG 波形、样本值或设备绝对时间戳。单位证据及其适用范围另见
`neuracle_realtime_unit_evidence.md`；本文件不取代该单位证据。

所有窗口都使用 4.0 秒长度和 0.5 秒步长。Marker 使用半开区间：
`window_start <= event_timestamp < window_end`。因此处于窗口结束边界的事件不属于该窗口；重叠窗口可以关联同一个真实事件。

## 20 秒连续性验证（无 Trigger）

- 约 19.9k 个连续 EEG 样本。
- 59 个 EEG 通道，1000 Hz。
- EEG 单位为 `uV`，单位证据等级为 `vendor_confirmed`。
- `expected_windows` 等于 `emitted_windows`，共 32 个 4 秒窗口。
- `failed_windows = 0`，`timestamp_gap_count = 0`，`buffer_overflow_count = 0`。
- 数据包连续性计数（missing、duplicate、out_of_order、malformed）均为 0。

## 30 秒 Trigger–Window 验证

- 按顺序接收五个真实事件码：1、2、3、4、20。
- 每个事件均关联到至少一个窗口。
- Marker 采用上述半开边界；重叠窗口可关联同一个真实事件。
- 窗口、数据包和 TimestampBuffer 指标均通过：未见数据包连续性异常、时间戳 gap 或 Buffer overflow。

## 60 秒稳定性验证

- 约 59.9k 个连续 EEG 样本。
- `expected_windows = emitted_windows = 112`。
- `failed_windows = 0`，`window_completion_rate = 1.0`。
- `contiguous_segment_count = 1`，`timestamp_gap_count = 0`。
- `buffer_peak_samples = 4000`，`buffer_overflow_count = 0`。
- 数据包连续性计数（missing、duplicate、out_of_order、malformed）均为 0。
- Probe 结束后的最终状态为 `state=stopped`、`connected=false`、`last_error=null`。
- `waveforms_saved=false`；未持久化 EEG 波形。

## 隐私边界

本验证记录不包含网络地址、串口标识、模块名称或类型、设备序列号（含哈希）、设备绝对时间戳、EEG 数值、受试者信息或本机绝对路径。

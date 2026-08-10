# Neuracle BDF Reader

`NeuracleBDFReader` 接收一个 BDF 文件路径和已验证的 `UnitEvidence`。单位证据必须是 model-safe，且规范化单位必须为 `uV`；文件不存在、没有可用 EEG 通道、annotation 越界或 metadata 不一致时会抛出 `ValueError` 或 `FileNotFoundError`。

Reader 使用 MNE 的惰性打开：`mne.io.read_raw_bdf(..., preload=False)`。随后为离线 trial 提取会调用 `get_data(..., start=0, stop=n_times, units="uV")` 读取完整连续 EEG；这不是 metadata-only 操作。读取结果不再进行额外的 `1e6` 缩放。

输出是 `RawEEGRecord`：

- `eeg` 为只读 `float32` 数组，形状为 `[C, T]`；
- 单位为 `uV`，由传入的 `UnitEvidence` 明确声明；
- 保留原始选中通道名、采样率、时间戳、通用 annotation 事件及来源 metadata。

仅保留 channel type 为 `eeg` 且名称不为 `ECG`、`HEOR`、`HEOL`、`VEOU`、`VEOL`（忽略大小写和首尾空格）的通道。被排除通道及全部原始通道信息会保存在 record metadata。

Reader 不做滤波、重采样、标准化、补通道、参考重设、窗口切片或模型调用。

当前标准输入为导出目录中合并连续信号与完整事件的 `1.bdf`。

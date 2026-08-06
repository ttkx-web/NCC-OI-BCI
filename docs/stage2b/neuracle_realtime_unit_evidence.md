# Neuracle JellyFish 实时单位证据（脱敏）

## 结论

- 确认对象：Neuracle Collect 通过 JellyFish TCP 输出的实时数据。
- 确认单位：生理信号数值为 µV；代码内部规范字符串为 `uV`。
- 证据等级：`vendor_confirmed`。
- 确认日期：2026-08-06。
- 证据类型：厂商书面确认。

## 适用范围

证据覆盖实时 TCP Data 中的模拟生理通道：EEG、EOG、HECG 和 ECG。它们在通道级
合同中映射为 `uV`。

Trigger 是事件码而非生理幅值，通道级合同将其映射为 `code`，绝不解释为 µV。未被
证据明确覆盖的通道类型映射为 `unknown`，不根据类型名称以外的信息或数值幅值猜测。

## 流水线边界

`JellyFish raw mixed stream → channel-aware unit contract → EEG-only selector → TimestampBuffer → 4 s windowing`

原始混合流包含 Trigger，因此 `unit=mixed` 且不能直接送入模型。只有 EEG-only selector
输出的全部 EEG 通道均具有 `uV` 和 `vendor_confirmed` 证据时，才具备进入后续流水线的
资格。TimestampBuffer 仅接收 EEG-only `uV` chunks；Windowing 仅生成不跨时间戳 gap 的
连续窗口。该 `model_safe` 仅表示单位合同满足，不表示已完成预处理、重采样、通道映射、窗口
完整性验证或模型输入契约。

本文不记录 IP、COM 号、设备序列号或哈希、厂商联系人、私密通信原文、受试者信息或原始
EEG 数据。

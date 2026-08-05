# Stage 2B Neuracle JellyFish 真实设备验证（脱敏）

测试日期：2026-08-05。

本次验证采用两台计算机：采集/转发端和 NCC-OI-BCI probe 客户端。本文不记录
IP 地址、模块名称、设备序列号（含哈希对应关系）、受试者信息、EEG 波形或幅值。

## 已验证结果

- META 已就绪。
- 共 65 个通道：59 EEG、1 ECG、4 EOG、1 Trigger。
- 采样率为 1000 Hz。
- 10 秒匿名探测：100 packets、10,000 samples。
- 30 秒匿名探测：299 packets、29,900 samples。
- 连续性计数均为 0：gaps、duplicate、out-of-order、malformed、missing。
- restart 与 reconnect 已通过。
- probe 退出后客户端 socket 已释放。
- 本次没有持久化 EEG 波形。

现场曾因两台计算机同时使用多个网络接口而发生应用层 META 失败。改为两台计算机
仅使用同一隔离局域网后，连接和 META 验证成功；本文不记录具体网络地址。

## 仍受安全约束的事项

- 实时 TCP float 的物理单位尚未确认，因此保持
  `raw_unit=unknown`、`unit_evidence_level=realtime_unverified`、
  `model_safe=false`。
- TCP Trigger 的实际行为尚未验证。
- 实时数据被安全阻断，不能进入 Preprocessor 或模型集成。

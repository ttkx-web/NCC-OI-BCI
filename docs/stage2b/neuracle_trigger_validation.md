# Stage 2B Neuracle Trigger 验证（脱敏）

## 现场结果

- USB DCP 设备查询成功。
- 周期 Trigger 使同步器事件灯闪烁。
- Collect 显示事件标记。
- Probe 收到 5 个事件，事件码顺序为 1、2、3、4、20。
- 每个事件均具有设备 `raw_timestamp`；本文不记录其具体数值。
- 相邻事件间隔约 2 秒。
- 连续性与解析计数均为 0：`missing`、`duplicate`、`out_of_order`、`malformed`。
- Probe 结束后为 `state=stopped`、`connected=false`、`last_error=null`。
- 未保存 EEG 波形。
- 实时单位仍为 `raw_unit=unknown`，且 `model_safe=false`。

本文不记录 IP、COM 号、设备序列号或其哈希、模块名称、受试者信息，亦不记录完整
设备原始时间戳或 probe 原始输出。

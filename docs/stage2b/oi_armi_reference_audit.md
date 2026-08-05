# oi-armi Neuracle JellyFish 参考实现审计

参考仓库：`https://github.com/ttkx-web/oi-armi`。

审计 commit：`61b7b855376814c1c427871d6d3d623e8e49e9e1`（`main`）。本次在系统临时目录做浅克隆只读检查，未将参考仓库复制到 NCC-OI-BCI。

## 已核对的文件

- `oi-mi/acquisition/neuracle_acquirer.py`：`NeuracleAcquirer`。
- `oi-mi/acquisition/base.py`：`AbstractAcquirer` 与其二元 `(samples, timestamps)` `EEGChunk` TypeAlias。
- `oi-mi/collect/neuracle_api.py`：`resolveMeta`、`resolveMetaEachModule`、`resolveData`、`resolveDataEachModule`、`ConnectState`、`DataServerThread`、`mergeMetaTriggerModule`、`isDataPacketLost`、`combineDataAndTrigger`。
- `oi-mi/collect/README.md`：当前主链路使用 `neuracle_api.py` 的说明。
- `oi-mi/config.yaml`：`device.neuracle_host=127.0.0.1`、`device.neuracle_port=8712`、示例 `sfreq=250`。

## 观察到的协议行为

- `DataServerThread.connect(hostname="127.0.0.1", port=8712)` 使用 TCP socket。
- `ConnectState` 定义 `NOTCONNECT`、`CONNECTED`、`READY`、`RUNNING`、`ABORT`。
- `resolveMetaEachModule` 解析 `moduleName`、`moduleType`、`serialNumber`、`channelCount`、`channelNames`、`channelTypes`、`sampleRates`、`dataCountPerChannel`、`maxDigital/minDigital`、`maxPhysical/minPhysical` 与 `gain`；同时还解析 `personName`。
- `resolveData` 解析 `startTimeStamp`、`timeStampLength`、`triggerCount`、`moduleCount` 及每模块 float 数据。代码以 `timeStampLength * sample_rate / 1000` 计算数据点数，因此这里的设备时间戳按毫秒解释是合理的协议候选，仍须在真实设备探测时验证。
- `DataServerThread` 对 bulk 转发直接使用数据模块；对 per-module 转发把 serial number `0` 的 Trigger META 合并，并由 `combineDataAndTrigger` 将 Trigger 放入组装缓冲。
- `isDataPacketLost` 比较前一包 `startTimeStamp + timeStampLength` 与下一包 `startTimeStamp`，用于发现可能丢包。

## 参考实现不直接继承的部分

`NeuracleAcquirer.get_chunk()` 与 `get_new_samples()` 用 `np.arange(...) / sfreq` 从零生成每次调用的局部时间戳，并以 `data[:n_channels]` 截断通道；未提供实时 TCP float 单位的显式证据。`DataServerThread.connect()` 使用循环/`print`，重连次数与错误收敛不足。

本项目的 Adapter 改为：保留完整 META 通道、以原始设备时间戳生成跨 chunk 时间轴、记录 raw timestamp 与匿名诊断、通过有限指数退避重连、并将 `raw_unit="unknown"` 固定为 `realtime_unverified` / 非 model-safe，直到可验证的实时–BDF 比例证据存在。

## 许可证阻断

`oi-mi/collect/neuracle_api.py` 的原始头部写明 `Copyright (c) 2022 Neuracle, Inc. All Rights Reserved.`；参考仓库未发现可确认的内部复用许可证。因此 NCC-OI-BCI 未复制其 TCP 或二进制协议解析代码。`src/bci_dayloop/vendor/neuracle/` 仅保留来源与边界说明；真实连接需要厂商授权的 backend 或许可确认后再实现。

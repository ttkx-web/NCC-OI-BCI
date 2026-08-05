# Neuracle JellyFish vendor boundary

本目录不包含任何 Neuracle JellyFish 协议解析主体。

参考来源：`ttkx-web/oi-armi`，commit `61b7b855376814c1c427871d6d3d623e8e49e9e1`。

审计的来源文件：

- `oi-mi/collect/neuracle_api.py`
- `oi-mi/acquisition/neuracle_acquirer.py`

`neuracle_api.py` 带有 `Copyright (c) 2022 Neuracle, Inc. All Rights Reserved.`
版权头，参考仓库中没有可确认的内部复用许可证。因此该文件及其协议解析代码没有被复制、改写或导入本项目。

`bci_dayloop.realtime.neuracle_jellyfish` 只定义 Adapter 合同：经授权的厂商 backend 负责 TCP/二进制协议和 META/Data 包解析；本项目负责实时合同校验、连续时间轴、匿名日志、单位阻断和有限重连。

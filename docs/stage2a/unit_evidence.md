# Unit Evidence

当前项目的 Reader 配置明确要求并使用 `uV`：`UnitEvidence` 的 model-safe 证据只能在规范化单位为 `uV` 时传给 `NeuracleBDFReader`，MNE 数值读取使用 `get_data(..., units="uV")`。

单位文字统一规则如下：

- `uV`、`µV`、`μV` 均规范化并记录为 `uV`；
- 其他当前支持的显式单位为 `V` 和 `mV`；
- 禁止根据 EEG 波形幅值、physical range 或常见经验推断文件单位。

`UnitEvidence` 区分 `unknown`、`header_candidate`、`vendor_confirmed`、`official_reader_verified` 与 `calibration_verified`。只有后面三种 model-safe 级别且存在规范化单位时才能作为模型输入的单位证据。

当前仓库未在本文档中主张任何厂商证明或校准证明。原始 NDF 元数据、厂商导出说明与 BDF physical-digital scaling 的交叉证据仍为**待补充**；在来源明确前，不应把未验证字段升级为 `vendor_confirmed` 或 `calibration_verified`。

# Unit Evidence

当前项目已正式确认 EEG 文件单位为 `uV`。`UnitEvidence` 使用 `raw_unit="uV"`、`normalized_unit="uV"` 与 `evidence_level="vendor_confirmed"`，因此 `is_model_safe=true`。MNE 数值读取使用 `get_data(..., units="uV")`。

单位文字统一规则如下：

- `uV`、`µV`、`μV` 均规范化并记录为 `uV`；
- 其他当前支持的显式单位为 `V` 和 `mV`；
- 禁止根据 EEG 波形幅值、physical range 或常见经验推断文件单位。

`UnitEvidence` 区分 `unknown`、`header_candidate`、`vendor_confirmed`、`official_reader_verified` 与 `calibration_verified`。只有后面三种 model-safe 级别且存在规范化单位时才能作为模型输入的单位证据。

本结论仅确认本次离线 Reader 输入/输出单位，不把波形幅值或 BDF physical range 当作单位证据，也不主张额外的校准证明。转换工具名称与版本若未由导出记录提供，会在 provenance 中显式记录为 `unverified` / 空值，而不会虚构厂商版本信息。

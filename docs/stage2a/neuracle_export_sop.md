# Neuracle BDF Export: Verified Facts

本文件仅记录本次验证过的导出结果，不推测厂商软件的未验证操作步骤。

- 导出目录可能包含 `data.bdf`、`evt.bdf` 和 `1.bdf`。
- `data.bdf` 包含连续信号，但没有 annotation。
- `evt.bdf` 是事件载体。
- `1.bdf` 同时包含连续信号和完整事件，应作为当前标准输入。
- 已验证样例为 64 通道、250 Hz、650 个事件，四个类别各 40 个 trial。

本文不包含本机绝对路径，也不虚构 NDF 转换按钮、命令、参数或厂商操作步骤。任何新增转换流程必须先由可复现的厂商文档或实际导出结果验证后再补充。

真实 BDF 文件、SHA256 体检报告以及 `runs/stage2a/` 下生成的 JSON 均不得提交到 Git。

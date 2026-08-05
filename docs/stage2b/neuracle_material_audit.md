# Stage 2B 博睿康资料只读审查

审查日期：2026-08-05。范围为 `E:\code\博睿康资料` 中的本机文件，以及为比较而读取的 NCC-OI-BCI 现有 vendor 边界和既有参考审计记录。本轮没有连接设备、启动 Collect/JellyFish、运行厂商安装包、解压安装包或二进制文件，也没有修改资料原件、Python 源码、模型、Runtime、Replay 或 Streamlit。

## 1. 审查范围

`manifest.csv` 覆盖 2,109 个文件，记录相对路径、扩展名、大小、修改时间、SHA-256、推断类别和审查优先级；文件内容不在清单中。厂商资料中可阅读的核心材料为三份 PDF、Collect 安装包的文件元数据、随附的 JSON 配置以及 NeuracleReader 离线读取示例。未发现厂商 JellyFish TCP 协议说明、JellyFish 独立手册、SDK API 手册或 Python 实时示例。

## 2. 安全与脱敏说明

报告不含密码、密钥、完整设备序列号、身份字段、受试者信息、摄像头凭据、EEG 波形或无关内部网络配置。配置文件中的端口仅在与数据服务直接相关时引用；任何序列号仅允许在现场产物中使用截断 SHA-256。META 中的身份字符串在本报告中一律记为 `[REDACTED]`。

## 3. 高价值文件清单

| 文件 | 证据性质 | 价值 |
| --- | --- | --- |
| `E:\code\博睿康资料\NeuroHUB多模态同步采集软件V2.11.pdf` | 厂商 Collect 手册 | 实时采集、数据转发、LSL、采样率与工作流 |
| `E:\code\博睿康资料\NeuroHUB 可穿戴式多模态研究平台V3.pdf` | 厂商产品手册 | NCA0002 型号、采样率、同步器与采集电脑关系 |
| `E:\code\博睿康资料\多模态设备同步器产品说明书V3.pdf` | 厂商同步器手册 | TriggerBox、DCP、同步器 TCP/串口接口（非 JellyFish） |
| `E:\code\博睿康资料\Neuracle应用软件\Recorder-TxCSSupport_Base_4200-20220513-6ba4ed99\Conf\SystemSetting.json` | 随附软件配置 | `DataServicePort: 8712` |
| 同目录 `Conf\Product\Neusen H\SystemSetting.json` | 产品配置 | 同为 `DataServicePort: 8712` |
| 同目录 `Conf\Product\Neusen M\SystemSetting.json` | 产品配置 | 同为 `DataServicePort: 8712` |
| `E:\code\博睿康资料\Collect2.11_202504\Neuracle_Collect_Release_2.11_b575d036.exe` | 安装包元数据，未执行 | Collect 交付包存在；228,852,850 B；PE FileVersion 9.5.3.0；SHA-256 前 16 位 `a21ff31929dff75a` |
| `E:\code\博睿康资料\NeuracleReader_v1\README.md` | 示例代码 README | 仅说明 BDF/NDF 离线读取，不是实时协议资料 |
| `src\bci_dayloop\vendor\neuracle\neuracle_api.py` | 授权复制的参考实现，非厂商手册 | TCP、META/Data 解析与协议候选；SHA-256 前 16 位 `eb9dd03343751b9f` |
| `docs\stage2b\oi_armi_reference_audit.md` | 既有项目审计记录，非厂商资料 | 记录参考仓库、commit 和历史审计边界 |

三个核心 PDF 的 SHA-256 前 16 位依次为：Collect 手册 `cf5f190cea4e95dc`、平台手册 `07b803f6eca8a9d0`、同步器手册 `ccc9ba839a86f4ea`。完整哈希见 manifest。

## 4. 软件和协议版本

- **CONFIRMED（厂商手册）**：Collect 手册扉页为《NeuroHUB 多模态同步采集软件》，说明书编制日期 2024-12-17，适用产品软件版本为 Collect V2.0 及以上兼容版本（Collect 手册 PDF 第 3 页，版本信息）。
- **CONFIRMED（文件名/元数据）**：本机还保存有 `Collect2.11_202504` 安装目录和 `Neuracle_Collect_Release_2.11_b575d036.exe`；这只能证明交付文件的命名版本，不能证明已安装或正在运行的版本。
- **CONFIRMED（厂商产品手册）**：平台手册列出 NCA0002 为 64 导脑电（平台手册 PDF 第 10 页，1.3 产品型号）。该手册列出的设备采样率为 1000/2000/4000/8000/16000 Hz（第 37 页，8.2 性能参数）。
- **UNKNOWN**：资料未给出 JellyFish 版本、JellyFish 协议版本、Collect 与 JellyFish 的版本兼容矩阵，也没有 NCA0002 专用 TCP/META 差异说明。

## 5. 两台电脑网络连接结论

| 问题 | 结论 | 依据与边界 |
| --- | --- | --- |
| JellyFish 是 TCP 服务端还是客户端？ | **INFERRED：服务端** | 授权参考实现的 `DataServerThread.connect()` 主动以 TCP 客户端连接“运行 JellyFish 的电脑”及其“opened port”（`neuracle_api.py:470-501`）。随附配置的键名为 `DataServicePort`，值为 8712（`SystemSetting.json:45`）。未找到厂商 JellyFish 手册，不能将该结论升级为厂商文档确认。 |
| NCC-OI-BCI 应主动连接还是监听？ | **CONFIRMED：主动连接** | NCC wrapper 调用 `server.connect(hostname=host, port=port)`（`vendor/neuracle/backend.py:39-49`），适配器配置也只提供 `host`/`port`（`realtime/neuracle_jellyfish.py:43-54`）。 |
| 默认端口是否为 8712？ | **CONFIRMED：随附配置的 DataServicePort 为 8712；INFERRED：它是 JellyFish 默认监听端口** | 三个同类 `SystemSetting.json` 都在第 45 行给出 8712；参考实现默认值亦为 8712（`neuracle_api.py:470-475`）。未找到厂商文档把该配置键明确命名为 JellyFish。 |
| 端口能否修改？ | **INFERRED：配置文件定义了可配置端口字段；现场不应直接改文件** | `DataServicePort` 是 JSON 值而非编译常量（`SystemSetting.json:45`），但没有厂商文档确认支持方式、重启要求或影响。仅可在厂商确认的界面/流程下变更。 |
| 是否支持远程局域网客户端？ | **UNKNOWN** | 客户端代码可传入任意 `host`（`neuracle_api.py:470-484`），这不证明服务端会监听非回环地址。 |
| 可否绑定回环、所有接口或指定网卡？ | **UNKNOWN（服务端 bind）** | 资料仅确认 NCC 客户端默认连接回环地址；没有 JellyFish bind 配置、监听地址或网络权限的厂商证据。 |
| 是否存在仅允许本机连接的版本/配置？ | **UNKNOWN** | 未发现此类开关或版本差异。 |
| 两台电脑是否需要额外授权/配置？ | **UNKNOWN（授权）；CONFIRMED（需要现场设置转发）** | Collect 手册要求在实时采集界面选择探头并点击数据转发开始（PDF 第 29 页，3.2.3.1.5）；未发现两机许可证说明。 |
| 防火墙应放行什么？ | **INFERRED：若现场证实 A 向局域网监听，则 A 入站 TCP 8712、B 到 A 的出站 TCP 8712** | 端口证据如上；服务端 bind 和远程可达性尚未确认，故该规则必须在现场网络确认后实施。 |
| 多个 TCP 客户端能否同时连接？ | **UNKNOWN** | `MaxConnectedDeviceCount: 6`（同一 JSON 第 44 行）是设备数量设置，不能推成六个数据服务客户端。 |
| 客户端断开后 JellyFish 是否继续转发？ | **UNKNOWN** | 客户端代码仅在读到空 socket 时自身进入 ABORT（`neuracle_api.py:605-614`），未描述服务端行为。 |
| 新客户端重连后是否重发 META？ | **INFERRED：客户端协议期待在每次连接后先收 META** | 解析器在 META 未完成时拒绝数据（`neuracle_api.py:735-789`），并在收到 META 后发确认；这不是厂商服务端重连保证。 |

## 6. JellyFish host / port / bind 结论

- **CONFIRMED**：NCC 的可配置客户端参数是 host 与 port；默认 host 为 `127.0.0.1`、port 为 8712（`realtime/neuracle_jellyfish.py:43-54`）。这只是客户端默认，不是厂商服务端绑定声明。
- **CONFIRMED**：随附软件配置中 DataServicePort 为 8712（上述三个 JSON 的第 45 行）。
- **UNKNOWN**：JellyFish 的进程名、实际监听者、bind 地址、端口修改入口、是否需管理员权限、远程访问限制和客户端并发数。
- **现场结论**：两机模式只能作为“候选配置”推进：电脑 A 运行 Collect 并在其界面启用数据转发；电脑 B 使用 A 的经现场确认的可达地址主动连接 8712。只有 `Test-NetConnection`、metadata-only probe 和短时 probe 都成功后，才能把该候选升级为现场已证实配置。

## 7. META 字段

**INFERRED（授权参考实现，非厂商协议手册）**：`resolveMeta()` 用小端序解包通用头，`resolveMetaEachModule()` 解析：`[REDACTED]` 身份字符串、模块名称、模块类型、序列号、通道数、通道名、通道类型、每通道采样率、每通道数据点数、数字最大/最小值、物理最大/最小值和 gain（`neuracle_api.py:210-305`）。

- 通用头使用 `unpack("<H4I", ...)`；模块的固定身份/名称字符串槽位为三段各 30 bytes，随后为两个小端无符号 32-bit 整数（`neuracle_api.py:217-224, 264-289`）。
- META 头/尾 token 候选为 `0x5FF5` / `0xF55F`；数据包候选为 `0x5AA5` / `0xA55A`（`neuracle_api.py:681-727`）。META 确认包候选为 `F55F5FF5`（第 781-784 行）。
- 通道数动态读取（`channelCount`），通道名与类型均按 10-byte UTF-8 槽位读取；不是固定 64 通道（第 264-304 行）。
- 厂商 Collect 手册仅在 GUI 中列出 EEG、EMG、EOG 三种通道类型（PDF 第 23 页，3.2.2.3.4 前的通道类型说明）；这不是 TCP META 的合法枚举全表。
- **UNKNOWN**：真实设备发出的 META 的字段版本、所有合法通道类型、字符串编码异常处理、NCA 系列专用字段，以及 META 是否在每次重新连接后一定重发。

## 8. Data packet 与 timestamp

**INFERRED（授权参考实现）**：数据头按小端序解为头长度、总长度、`startTimeStamp`、`timeStampLength`、触发数、flag、模块数（`neuracle_api.py:322-367`）。payload 数据按小端 IEEE-754 32-bit float 读取（第 370-398 行）。

- 连续性检查用“上一包起始时间 + 长度”与“下一包起始时间”比较（`isDataPacketLost()`，第 892-906 行）；这能发现缺口候选，但不能独立区分丢包、重启、重复或乱序。
- bulk 与 per-module：flag 偶数走 bulk 直接取数据模块；flag 奇数走 per-module，序列号 0 被当作 Trigger 模块并与数据组装（第 803-842、908-1005 行）。
- 参考实现以 `timeStampLength * sample_rate / 1000` 计算点数（第 924-925 行），因此“毫秒”是合理的实现假设；但资料没有明确给出时间戳单位、复位语义、序号、5 ms 周期或精确 packet-to-sample 合约。
- **UNKNOWN：资料未明确支持，需真实设备实验确认。** timestamp 是否重连/重新开始采集后归零、是否严格跨包连续、是否有序号、如何判定 duplicate/out-of-order、5 ms 周期以及所有时间戳单位。

## 9. 通道与采样率

- **CONFIRMED（厂商 Collect 手册）**：Collect GUI 可选 250、500、1000、2000、4000 Hz；支持正常采集、短路噪声、自检方波。多设备时降采样仅影响实时显示（250 Hz、前 8 通道），本地存储仍按设定采样率（Collect 手册 PDF 第 20 页，3.2.2.3）。
- **CONFIRMED（厂商平台手册）**：NCA0002 是 64 导脑电（平台手册 PDF 第 10 页，1.3）。
- **UNKNOWN**：NCA0002 在 JellyFish 里是否恰为 64 个 TCP 数据通道、是否附加 Trigger、以及当前现场 4000 Hz 设定是否有专用转发差异。现场必须以 META 实测通道数、通道名、通道类型和 sampleRates 为准。

## 10. 实时单位与缩放

`raw_unit = unknown`；`unit_evidence_level = realtime_unverified`；`model_safe = false`。

**INFERRED（授权参考实现）**：数据负载按 float32 读取，META 中有数字/物理范围和 gain 字段（`neuracle_api.py:258-305, 370-398`）。但没有找到厂商 JellyFish 协议文档说明该 float 已是 uV、V、nV、digital count，或给出 `physical = f(digital, gain)` 公式。

平台手册的系统噪声 `≤1 uVrms`（PDF 第 37 页，8.2）和 Collect 截图的显示单位不能证明 TCP float 单位。BDF 与实时 TCP 数值是否一致、不同型号缩放是否不同、NCA0002/64 通道/4000 Hz 的专用缩放均为 **UNKNOWN：资料未明确支持，需真实设备实验确认。** 在有同一时间段 BDF-实时对齐证据前，不得将实时数据输入模型。

## 11. Trigger、同步盒与 LSL

- **CONFIRMED（厂商平台手册）**：NCA0002 标配 Trigger 转发线；平台包含个人、多人和公共事件转发器（平台手册 PDF 第 12、15 页，2.2、2.5）。同步器将带时间戳的数据通过 Wi-Fi 或有线方式传给电脑（第 15 页，2.4）。
- **CONFIRMED（厂商同步器手册）**：TriggerBox 可经 Micro-USB 在 PC 上形成 COM 口；DCP 串口参数为 115200、8 数据位、1 停止位、无校验、无流控（同步器手册 PDF 第 23-25 页，3.5）。该手册还描述同步器的 TCP 4321 接口，但它针对同步器下设备/光电池/TriggerIN/OUT，不是 JellyFish 8712（第 20、31 页）。
- **CONFIRMED（厂商 Collect 手册）**：实验概要区有 LSL trigger 开关（PDF 第 16 页，3.2.2.1）；LSL 设置说明该勾选开启/关闭 LSL 数据转发（第 23-24 页，3.2.2.3.4）。
- **INFERRED（授权参考实现）**：per-module 模式将序列号 0 作为 Trigger 模块；bulk 模式把最后通道按 Trigger 重采样，二者逻辑不同（`neuracle_api.py:803-842, 871-1005`）。
- **UNKNOWN**：Trigger 是否一定在 JellyFish TCP 中转发、LSL marker 与设备时间戳是否同钟、LSL 是否独立于数据转发、硬件 Trigger 的优先级，以及每个 Trigger 精确映射到 EEG sample 的官方规则。现场优先采用硬件 Trigger/同步器并用短时 probe 验证；不要假设 LSL 和设备时钟天然同步。

## 12. 断线与重连

**INFERRED（项目代码）**：NCC adapter 在 `disconnect()` 时清空本地 metadata、事件与时间戳状态（`realtime/neuracle_jellyfish.py:142-160`），随后可有限次重连；新连接必须重新通过 metadata readiness（第 139-168、232-267 行）。vendor TCP 读取到空 socket 后只将本地状态置为 ABORT（`neuracle_api.py:605-614`）。

**UNKNOWN**：厂商服务端对客户端断开、并发客户端、重新连接、META 重发、采集不中断和 packet sequence 的实际行为。现场 probe 应把“断线后立即停止、重新 metadata-only、再 10 秒短探测”作为独立验收项目。

## 13. 与 oi-armi / vendor backend 的比较

- **不完全相同（CONFIRMED）**：当前 vendor 文件包含 NCC 专用的有界 update queue、带原始时间戳的 update item 和线程收尾；这些增量在 `vendor/neuracle/README.md` 的 “NCC-OI-BCI minimal changes” 中逐项列出，并可见于 `neuracle_api.py:458-464, 522-547, 844-869`。
- **协议主体声称保留（INFERRED）**：同一 README 声明原始协议常量、二进制解析和字段含义保持；既有 `oi_armi_reference_audit.md` 记录参考路径为 `oi-mi/collect/neuracle_api.py`、commit `61b7b855376814c1c427871d6d3d623e8e49e9e1`。
- **无法做字节级/函数级原始比对（UNKNOWN）**：该 reference commit 不在本地 Git object database，资料目录也没有候选原始 `neuracle_api.py`；本轮不得联网获取或上传资料。因此不能声称“仅格式/注释差异”，也不能排除原始协议字段差异。
- **当前 backend 的适用性（INFERRED）**：其接口适用于参考实现所述的 TCP/META/Data 版本，但不能证明适用于本机 Collect 交付版本或 NCA0002。必须以 metadata-only、10 秒无波形持久化 probe 和单位对齐实验验证。
- **NCA0002 专用差异（UNKNOWN）**：厂商资料只确认其为 64 导型号，没有协议专用字段或 parser 分支资料。

## 14. NCA0002 的适用性

**CONFIRMED**：NCA0002 为 64 导脑电；设备手册列出 Wi-Fi 传输到采集软件（平台手册 PDF 第 10、12 页）。

**UNKNOWN**：其现场探头数、是否含 Trigger 通道、实际采样率、JellyFish forwarding 方式、单位缩放和 TCP 可达性。当前配置应把这些列为期望值/验收项，不能硬编码为已确认事实。

## 15. 可直接用于现场操作的步骤

只可在另行获准的现场操作轮次执行 `two_pc_probe_draft.md`。关键规则是：A 先完成采集与转发，B 再连接；先 metadata-only，再短时 10 秒 probe；探测输出不保存波形且模型保持禁止。

## 16. 仍需真实设备验证的问题

1. JellyFish 是否监听电脑 A 的局域网地址，以及实际 bind/端口修改入口。
2. Windows 防火墙实际命中方向与客户端并发上限。
3. 每次连接、重连、开始采集后 META 的发送行为。
4. timestamp 的单位、重置、连续性、序号和 5 ms 周期。
5. NCA0002 64 通道、Trigger、4000 Hz 的 META 与转发模式。
6. TCP float 的单位、物理/数字/gain 公式及 BDF-实时一致性。
7. Trigger 在 TCP 与 LSL 中的出现、同钟关系和 sample 映射。
8. Collect/JellyFish 版本兼容性及两机授权要求。

## 17. 风险与阻塞项

- 远程 bind 证据缺失：不能预先创建宽泛防火墙规则或声称两机可用。
- 实时单位未验证：禁止模型推理和任何临床/实验结论。
- 协议原件缺失：vendor 的协议解释只能标为参考实现推断。
- 资料中存在安装包和二进制：本轮只记录元数据/哈希，未打开或执行。
- UI 按钮以官方手册中文名称为准；任何未在手册中确认的按钮、地址和端口均须“需现场界面确认”。

## 18. 引用索引

- 厂商官方：`E:\code\博睿康资料\NeuroHUB多模态同步采集软件V2.11.pdf`，PDF 第 3、16、20、23-24、29-30 页。
- 厂商官方：`E:\code\博睿康资料\NeuroHUB 可穿戴式多模态研究平台V3.pdf`，PDF 第 10、12、15、37-38 页。
- 厂商官方：`E:\code\博睿康资料\多模态设备同步器产品说明书V3.pdf`，PDF 第 20、23-25、31-33 页。
- 随附配置：`E:\code\博睿康资料\Neuracle应用软件\Recorder-TxCSSupport_Base_4200-20220513-6ba4ed99\Conf\SystemSetting.json:42-50`，及两个产品同名配置文件第 45 行。
- 参考实现（非厂商手册）：`src/bci_dayloop/vendor/neuracle/neuracle_api.py:210-305, 322-398, 470-501, 668-842, 871-1005`。
- 项目 wrapper（非厂商手册）：`src/bci_dayloop/vendor/neuracle/backend.py:39-96, 119-155`、`src/bci_dayloop/realtime/neuracle_jellyfish.py:43-54, 139-220`、`src/bci_dayloop/vendor/neuracle/README.md`。


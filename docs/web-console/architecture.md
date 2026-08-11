# NCC BCI Web Console 架构

## 1. 目标与 P0 范围

Web Console 是 NCC-OI-BCI 的长期维护控制面。P0 交付橙白中文界面、Runtime Package 只读注册表、数据集元数据发现，以及可替代 Streamlit 主要运行控制能力的 Replay REST/WebSocket 链路。Live 设备、实验执行、NeuroOnline、Rest-Tune、CBraMod 执行和 AR/Robot 集成都不在本阶段范围内。

## 2. Next.js / FastAPI 架构

调用链固定为：

```text
Next.js App Router
  -> FastAPI /api/v1 与 /ws/v1
  -> console-api services
  -> src/bci_dayloop
```

React 只负责交互和呈现。FastAPI 路由只负责验证、HTTP 状态码和 schema；服务层负责资源发现与运行编排；所有预处理、通道映射、重采样、滑窗、模型准备和推理由 `src/bci_dayloop/` 完成。

## 3. 目录结构

```text
apps/
├── console-web/          # Next.js + React + TypeScript
│   └── src/
│       ├── app/          # 页面路由
│       ├── components/   # 布局、统一 UI、状态组件
│       ├── lib/          # REST/WebSocket 客户端与格式化
│       └── types/        # API 事件类型
└── console-api/          # FastAPI
    └── app/
        ├── api/routes/   # 版本化 HTTP 路由
        ├── schemas/      # Pydantic 合同
        ├── services/     # Runtime、模型、数据集和 Run 编排
        └── websocket/    # Run 事件流
```

原 `web/app.py` 与 `web/ui_runtime.py` 保留为回归参考，不被新 API 导入。

## 4. REST API 合同

所有 REST 接口使用 `/api/v1`：

- `GET /health`：API 与 Runtime 可用性。
- `GET /system/status`：设备、Runtime 与 CUDA 摘要。
- `GET /models`、`GET /models/{model_id}`：Runtime Package schema v2 只读元数据，支持 `backbone`、`subject`、`adaptation` 筛选。
- `GET /datasets`、`GET /datasets/{dataset_id}`：匿名化 HDF5 元数据。
- `GET /subjects`：匿名被试 ID。
- `POST /runs/replay`：创建回放。请求只能设置数据集、被试、Session、模型、计算设备、速度、最大窗口和置信度阈值。
- `POST /runs/{run_id}/stop`、`POST /runs/{run_id}/restart`：直接控制现有 `PipelineController`。
- `GET /runs`、`GET /runs/{run_id}`：统一 Run 读模型。

`window_sec`、`step_sec`、目标采样率和目标通道不属于 Replay 请求，必须取自选中的 Runtime Package。

## 5. WebSocket 合同

Run 事件流位于 `/ws/v1/runs/{run_id}`，统一信封为：

```json
{
  "type": "prediction",
  "run_id": "run_123",
  "timestamp": 1786460000.123,
  "payload": {}
}
```

P0 白名单事件为 `state`、`prediction`、`latency`、`runtime_health`、`input_contract`、`error`。致命错误必须带 `fatal: true`，前端立即清空当前预测并显示“已阻断”，不得继续把旧输出当作有效命令。

## 6. Runtime source of truth

`src/bci_dayloop/` 是唯一 BCI Runtime source of truth。Replay 服务复用：

```text
load_runtime_package
  -> SlidingWindowDecoder
  -> PipelineController
  -> ReplayAcquirer
```

服务层不实现 `RuntimeModel.prepare`、`predict_prepared`、通道补全、重采样、滑窗或状态机。模型元数据只读取 `model_packages/**/package.yaml` 和 `runs/**/package.yaml` 的 schema v2，并生成不包含路径的稳定 `model_id`。

## 7. 橙白 UI 规范

界面使用暖白画布、白色大圆角卡片和橙色品牌强调。橙色用于主按钮、当前导航和重点数值；绿色表示正常与 SAFE，蓝色表示 RUNNING，黄色表示警告，红色表示失败或 BLOCKED，灰色表示空闲或禁用。禁止大面积暗色、霓虹、医院蓝和复杂渐变。设计 token 集中在 `globals.css`。

桌面端采用 Header、Sidebar、12-column 内容区和底部 Status Bar；优先适配 1440–1920 px，并保证 1280 px 可用。窄屏仅保证布局不破坏。

## 8. 中文界面规范

导航、页面标题、按钮、表头、错误提示和运行说明以中文为主。API path、模型名、技术状态值、JSON key、Runtime Package 字段和正式数据集名称可保留英文。代码标识符保持英文。

## 9. 隐私边界

浏览器和 API 不得获得真实姓名、身份证明、设备序列号、本机绝对路径、私有网络地址或原始脑电数据。模型与数据集服务内部可以持有 `Path` 用于加载，但 Pydantic 响应模型没有路径字段。服务对初始化错误使用安全文案，避免把底层异常中的路径发送给客户端。

`data/raw`、缓存、处理数据、checkpoint、run、model package 和 registry 均继续由 `.gitignore` 隔离。

## 10. 不显示 EEG 波形

Web Console 明确不显示 EEG 波形。Dashboard、Replay、Live 和 System 都只显示设备元数据、窗口事件、预测、延迟、合同与健康状态。浏览器不接收用于绘图的原始样本或降采样预览，WebSocket 也没有相应事件类型。自动测试递归检查事件 key，禁止 `samples`、`raw_eeg`、`waveform` 和 `waveform_preview` 数据字段。

## 11. Streamlit legacy 关系

`web/app.py` 和 `web/ui_runtime.py` 暂时原样保留，作为 Replay 行为回归参考。FastAPI 不依赖 Streamlit、session state 或 Streamlit UI 回调。待 React Replay 完成真实模型验收后，再单独决策 legacy 生命周期。

## 12. P1 Live 计划

P1 应在现有 `realtime_service` 边界内接入 JellyFish 与 Stage 2B，不修改 Stage 2B 输入合同。实现顺序为：连接生命周期、六项实时 Gate、packet continuity 指标、窗口/预测/命令事件时间线、fail-closed 验证，最后再开放 Live 控制。仍然不向浏览器发送原始脑电数据。

## 13. 后续扩展点

- Models：Package provenance、只读指标比较和选择 Replay。
- Experiments：后台评估任务、统一指标表和运行记录。
- Personalization / NeuroOnline：在 Checkpoint F 后增加独立配置与审计 UI。
- OI / AR：只消费经过阈值与安全 Gate 的 command 事件。
- CBraMod：待主运行时原生支持后，由 Package loader 暴露能力，Console 不自行加载实现。

在 Checkpoint B（Replay parity with Streamlit）通过验收前，不扩展以上功能。

## 本地启动

```powershell
python -m pip install -r requirements-console.txt
cd apps/console-api
uvicorn app.main:app --reload
```

另开终端：

```powershell
cd apps/console-web
pnpm install
pnpm dev
```

默认前端地址为 `http://127.0.0.1:3000`，API 为 `http://127.0.0.1:8000`。可通过 `NEXT_PUBLIC_CONSOLE_API_URL` 修改 API 地址。


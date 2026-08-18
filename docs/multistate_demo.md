# Omni Neural Decoder：EEG 多状态解码 Demo

这是一个独立的 Streamlit 可视化层，用于产品演示和视频拍摄。它读取真实 EEG HDF5 trial 的连续窗口，展示传统 EEG 特征生成的 Neural State Index、波形、PSD、皮层活动图、信号质量、延迟与规则式中文解读；运动意图 decoder contract 仍保留，但当前主界面默认隐藏其卡片。

它不修改生产 Runtime、Replay pipeline、NeuroOnline、Rest-Tune、50M / LaBraM / CBraMod 后端、正式训练或评测脚本。选择真实运动意图模型时，Demo 会通过既有 `load_runtime_package(...)` 和 `RuntimeModel` 接口调用对应 Package；不会自行加载 checkpoint 或重写预处理。

## 启动

从仓库根目录运行：

```bash
streamlit run web/multistate_demo.py
```

指定 EEG 数据路径和展示设备名：

```bash
streamlit run web/multistate_demo.py -- --data-path data/processed/bnci2014_001/subject_01.h5 --device cpu
```

页面侧栏可选择 HDF5 路径、session、单个 trial 或“全部 Trial”、窗口长度、推进步长和 0.5x–4x 数据流速度。选择“全部 Trial”后，点击一次“开始 / 继续”会连续播放当前 session 的所有 trial；trial 边界会自动推进，State、Motor Intent 和 latency history 保持连续。暂停会停在下一个待处理 window 前，重置会回到所选范围的第一个 trial 与第一个 window。数据应使用仓库标准 HDF5 格式（`data [N,C,T]`，并带有 `sample_rate`、`channel_names`、`unit` 等元数据）；BNCI2014_001 可用 `scripts/prepare_bnci2014_001.py` 生成。

无需启动 GUI，也可对一个真实窗口执行 smoke test：

```bash
python scripts/run_multistate_demo.py --data-path data/processed/bnci2014_001/subject_01.h5 --session 0train --trial 0
```

## 当前指标

每个值都做了 0–100 映射和指数平滑，仅供可视化：

| Neural State Index | 当前传统 EEG 特征 |
| --- | --- |
| 神经激活度 | RMS / 总活动量的对数映射 |
| 皮层唤醒度 | Beta / Alpha |
| 神经放松度 | 相对 Alpha 功率 |
| 认知参与度 | Beta / (Alpha + Theta) |
| 认知负荷 | Theta / Alpha |
| 注意稳定度 | 最近参与度窗口的变异系数 |
| 神经复杂度 | 五个频带功率分布的 spectral entropy |
| 神经同步度 | 最多 32 个通道的平均绝对相关系数 |

频带定义集中在 `bci_dayloop.demo.signal_features.EEG_BANDS`：Delta 1–4 Hz、Theta 4–8 Hz、Alpha 8–13 Hz、Beta 13–30 Hz、Gamma 30–45 Hz。PSD、每通道相对频段功率、皮层活动图输入、信号质量与延迟均由同一窗口计算，不会为皮层活动图重复执行 FFT。

## 皮层活动图

页面中的“皮层活动图”是 **sensor-derived cortical activity visualization**：它将当前 scalp EEG 的每通道 Theta + Alpha + Beta 相对功率，通过固定的 channel-to-template pixel mapping 叠加到预生成的左右半球静态 PNG 上。它用于 Demo visualization，**不是**经过 forward/inverse modeling 得到的 EEG source localization。

静态模板位于 `src/bci_dayloop/demo/assets/cortical/`，运行 Streamlit 时只会读取 PNG 并以 NumPy 叠加预计算 Gaussian masks；不会导入 Nilearn 或 MNE 来渲染皮层。开发者需要重新生成模板时才运行：

```bash
python tools/multistate_demo/generate_cortical_template.py
```

该工具使用 Nilearn 的 `fsaverage5` pial/sulcal surface 生成固定 512×384、透明背景的左右 lateral PNG，因此 Nilearn 仅是一次性开发依赖，不是 Demo runtime 依赖。

信号质量为展示用启发式分数，结合有效通道比例、平坦通道、极端振幅和高频噪声代理项；不是临床伪迹检测。

## Motor Intent

侧栏的“运动意图模型”提供两种实现：

- **Model Package**（发现可用 Package 时为默认）：自动扫描 `model_packages/`、`runs/` 和 `checkpoints/` 下 schema-v2 的四分类 motor-imagery Package。加载通过既有 `load_runtime_package(...)`，并让 `RuntimeModel.predict(RawEEGWindow)` 完成 Package 原有的通道映射、单位转换、重采样和输入 transform。Package 的窗口长度会自动成为 Demo 的推理窗口。
- **Demo Decoder**：`DemoMotorIntentDecoder` 不读取 trial label，而是根据频带特征生成稳定、分段主导且平滑过渡的左手、右手、双脚、舌部概率，便于无 Package 时快速展示。

两种实现都输出相同的 `label`、`label_cn`、`confidence` 与四分类 `probabilities`。Package 或设备切换会暂停数据流并只加载一次新 decoder；连续 trial、暂停和继续不会重复加载模型。

## 后续接入其他真实模型

Streamlit 页面只读取 `BrainStateResult`。接入真实 Workload、Emotion、Attention 或 Motor Intent decoder 时：

1. 在独立 decoder 中实现相同的 `decode(...) -> BrainStateResult`，或让 `MotorIntentDecoder.predict(...)` 返回同样的 `label`、`label_cn`、`confidence`、`probabilities` 结构；
2. 用真实 task head 的 0–100 分数替换相应 `states` 项；
3. 保持字段与 UI 不变，页面无需重写。

> Neural State Index 当前用于 Demo visualization，并非医学指标，也不是经过 benchmark 验证的多任务脑状态 decoder。AI 解读是规则生成的研究演示文字，不构成医疗或诊断结论。

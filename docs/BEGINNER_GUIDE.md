# BCI DayLoop 小白入门指南

这份文档用尽量少的专业术语解释：刚刚创建了什么、为什么这样设计、目前完成到哪里，以及你下一步应该输入什么命令。

项目位置：

```text
E:\code\BCI_DayLoop
```

它是一个独立项目，不会修改下面两个已有项目：

```text
E:\code\rest_tune_base
E:\code\oi-armi copy
```

## 一句话理解整个项目

这个项目把“想象左手、右手、脚、舌头”的脑电信号，逐步变成小车命令：

```text
BNCI2014_001 数据
    ↓
标准 HDF5 文件
    ↓
统一脑电预处理
    ↓
LaBraM Base 提取特征
    ↓
缓存 embedding
    ↓
只训练一个很小的 Linear 分类头
    ↓
离线模拟实时脑电
    ↓
网页显示预测和命令
```

命令映射如下：

| 脑电类别 | 小车命令 |
|---|---|
| `left_hand` | `LEFT` |
| `right_hand` | `RIGHT` |
| `feet` | `FORWARD` |
| `tongue` | `STOP` |

如果模型的置信度低于阈值，命令会强制变成 `STOP`。这是安全设计：不确定时不让小车继续运动。

## 刚刚已经做了什么

### 1. 创建了完整项目骨架

项目包含这些主要区域：

```text
configs/       配置文件
src/           正式 Python 源码
scripts/       可以直接运行的命令行脚本
web/           Streamlit 网页
tests/         自动化测试
data/          数据文件和缓存
checkpoints/   LaBraM 权重放置位置
runs/          训练输出和模型包
```

这样分目录的原因是：数据处理、模型、回放和网页互相分开。以后修改网页，不需要把模型代码全部重写；修改预处理，也能同时影响训练和回放，避免两边使用不同规则。

### 2. 下载并整理了 BNCI2014_001 Subject 1

已经通过 MOABB 获取了 Subject 1 的两个 session：

- `0train`：第一个 session，训练和验证使用
- `1test`：第二个 session，最终测试和回放使用

已经生成：

```text
data\processed\bnci2014_001_s01.h5
```

这个文件的内容是：

```text
data:        [576, 22, 1000]
labels:      [576]
subject_ids
session_ids
trial_ids
```

含义是：576 个试次、22 个 EEG 通道、每个试次 1000 个采样点。原始采样率是 250 Hz，所以每个试次是 4 秒。

为什么保存成 HDF5？因为 HDF5 适合保存大数组，Python、MATLAB 和其他工具都能读取，并且可以保存数据集属性，例如通道名、采样率和类别名。

### 3. 写了统一预处理

训练和回放都使用同一个 `EEGPreprocessor`，步骤是：

1. 删除 EOG、ECG、EMG、刺激标记等非 EEG 通道
2. 0.1–75 Hz 带通滤波
3. 50 Hz 陷波滤波
4. 重采样到 200 Hz
5. 从 V 转为 µV
6. 每个窗口、每个通道独立 Z-score
7. 变成 LaBraM 需要的 `[B,C,A,200]`

最后的 `200` 表示每个 patch 有 200 个采样点，也就是 200 Hz 下的 1 秒。

这样做的关键原因是：如果训练时的处理方式和回放时不一样，模型在测试时看到的信号分布就会改变，预测通常会变差。

### 4. 实现了 LaBraM Base + Linear 分类头

项目内置了一个只依赖 PyTorch 的 LaBraM Base 结构，接口名称是：

```text
labram-linear
labram_base_patch200_200
```

设计为两部分：

- LaBraM Base：冻结，不更新参数，只负责把脑电变成 embedding
- Linear head：只有最后一个小的 `nn.Linear` 会训练

为什么不直接训练整个 LaBraM？因为 RTX 4060 只有有限显存，而且一天内完成目标更适合先固定大模型、只训练小分类头。这样训练更快，也更不容易过拟合。

embedding 会缓存到：

```text
runs\day1_bnci_s01\embedding_cache\
```

缓存后，重复训练分类头不需要重新跑整个 LaBraM。

### 5. 实现了离线伪实时回放

`ReplayAcquirer` 会把 HDF5 中的第二个 session 当成一条连续 EEG 流，按照设置的速度一小块一小块地输出。

它支持：

- session
- speed
- loop
- window_sec
- step_sec

例如窗口 4 秒、步长 0.5 秒，意味着每隔 0.5 秒取最近 4 秒 EEG 做一次预测。

### 6. 实现了实时滑窗解码器

解码器会输出：

```text
prediction   预测类别
confidence   最大类别概率
latency_ms   本次预处理和推理耗时
command      小车命令
```

它不连接真实 EEG 设备，也不连接真实小车，只做离线回放。这是刻意的范围控制，先验证算法链路，再考虑硬件。

### 7. 实现了标准模型包

正式训练成功后，会生成：

```text
runs\day1_bnci_s01\model_package\
    head.pt
    model.yaml
    preprocessing.yaml
    label_map.json
    command_map.json
    metrics.json
    base_model.json
```

其中：

- `head.pt`：训练好的 Linear 分类头
- `model.yaml`：模型结构和通道顺序
- `preprocessing.yaml`：推理时必须复用的预处理参数
- `label_map.json`：数字类别和类别名称的对应关系
- `command_map.json`：类别和小车命令的对应关系
- `metrics.json`：验证集和测试集指标
- `base_model.json`：LaBraM checkpoint 路径、哈希和加载报告

加载模型包时会重新创建模型，不依赖原来的训练进程。这个行为已经用独立 Python 进程做过 smoke test。

### 8. 实现了 Streamlit 网页

网页位于：

```text
web\app.py
```

它可以选择数据文件、模型包、CPU/CUDA、置信度阈值、回放速度和最大窗口数，并显示：

- 当前预测
- 当前置信度
- 小车命令
- 当前延迟、平均延迟、P95 延迟
- EEG 波形
- 预测历史

网页中的模型和采集器名称来自 Factory 自动发现，不需要手工改网页列表。

## 目前完成到哪里

已经完成并验证：

- BNCI2014_001 Subject 1 HDF5
- 训练/回放共用的预处理
- LaBraM 随机初始化前向
- Linear adapter
- embedding 缓存接口
- 模型包保存和独立进程加载
- 离线 ReplayAcquirer
- 滑窗解码和 STOP 安全阈值
- Streamlit 页面
- 11 个自动化测试
- `python -m compileall .`

目前还没有完成正式训练，原因只有一个：本机还没有官方 LaBraM Base checkpoint。

## 下一步应该怎么做

### 第一步：创建环境

在 PowerShell 中输入：

```powershell
Set-Location E:\code\BCI_DayLoop
conda env create -f environment.yml
conda activate bci-dayloop
python -m pip install -e .
```

如果环境已经存在，可以跳过 `conda env create`，直接执行：

```powershell
conda activate bci-dayloop
python -m pip install -e .
```

### 第二步：放入 LaBraM checkpoint

把官方 LaBraM Base 权重放到：

```text
E:\code\BCI_DayLoop\checkpoints\labram-base.pth
```

如果文件名不同，需要修改：

```text
configs\day1_bnci_s01.yaml
```

里面的：

```yaml
model:
  checkpoint: checkpoints/labram-base.pth
```

不要用随机初始化权重做正式实验。`--random-init` 只用于检查代码线路是否接通。

### 第三步：运行一键流程

```powershell
Set-Location E:\code\BCI_DayLoop
conda activate bci-dayloop
python scripts\run_pipeline.py --config configs\day1_bnci_s01.yaml
```

脚本会自动：

1. 检查 HDF5 是否存在
2. 读取第一个 session
3. 切分训练集和验证集
4. 提取并缓存 LaBraM embedding
5. 训练 Linear head
6. 用第二个 session 做最终测试
7. 保存模型包
8. 在新的 Python 进程重新加载模型包

### 第四步：检查数据

```powershell
python scripts\inspect_dataset.py data\processed\bnci2014_001_s01.h5
```

你应该看到两个 session 和四个类别的数量。

### 第五步：离线回放

```powershell
python scripts\replay_offline.py `
  --config configs\day1_bnci_s01.yaml `
  --max-windows 20
```

每一行会输出一个 JSON，重点观察：

```text
prediction
confidence
latency_ms
command
```

### 第六步：启动网页

```powershell
streamlit run web\app.py
```

浏览器打开 Streamlit 显示的本地地址，通常是：

```text
http://localhost:8501
```

先选择数据文件和模型包，再点击 `Start replay`。

## 常见问题

### 报错：LaBraM Base checkpoint is missing

说明 checkpoint 没有放在配置指定的位置。检查：

```powershell
Test-Path E:\code\BCI_DayLoop\checkpoints\labram-base.pth
```

结果应该是 `True`。

### CUDA 不可用怎么办？

可以先使用 CPU：

```powershell
python scripts\smoke_test_labram.py --device cpu --random-init
```

网页也可以选择 `cpu`。正式训练建议使用 CUDA，并把 `embedding_batch_size` 保持为 4。

### 为什么网页提示没有 model package？

因为 Linear head 只有在正式训练后才会生成。先完成 checkpoint 放置和 `run_pipeline.py`。

### 为什么不直接接真实脑电设备？

当前目标是先验证数据、模型、回放、延迟和命令映射。真实设备需要额外的 SDK、网络连接、权限和安全测试，因此暂时没有加入。

### 为什么低置信度时是 STOP？

这是为了避免不确定预测直接控制小车。你可以在网页或 YAML 中调整阈值，但降低阈值会增加误动作风险。

## 目前没有实现的内容

以下内容是明确排除在一天目标之外的：

- 真实脑电设备
- Unity 或真实小车
- LoRA
- LaBraM 全参数微调
- Kubernetes
- MLflow
- 数据库
- React/FastAPI

等离线链路稳定后，再逐项扩展会更安全。


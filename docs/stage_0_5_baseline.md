# 阶段 0.5 基线记录

## 范围与版本

- 记录日期：2026-07-27（Asia/Shanghai）
- 当前分支：`chore/stage-0.5-baseline-contract`
- Git HEAD：`33a6b19d1fb977461aba23660fb7a39274121e9e`
- 本次范围：仅记录阶段 0.5 基线；未接入 50M，未合并其他分支，未修改运行代码。

## 固定实验参数

| 项目 | 固定值 |
| --- | --- |
| 数据集与受试者 | BNCI2014_001 Subject 1 |
| 会话划分 | `0train` / `1test` |
| 任务 | 四分类运动想象 |
| 类别 | `left_hand`、`right_hand`、`feet`、`tongue` |
| 历史 LaBraM Pipeline 窗口长度 | 4 秒 |
| 历史 LaBraM Pipeline 窗口步长 | 0.5 秒 |

## 阶段 0.5 窗口决策

### 历史 LaBraM 基线

- 原 LaBraM Pipeline 使用 4 秒窗口。
- 步长为 0.5 秒。

### 当前正式 50M 决定

- 阶段 0.5 的 50M Pipeline 使用 10 秒窗口，步长保持 0.5 秒。
- 目标采样率为 100 Hz；标准输入形状为 `64 × 1000`。
- patch 长度为 100 samples；时间 patch 数为 10；token 数为 640。
- embedding 维度为 512；Transformer 深度为 12；`output_layer_idx=8`；`aggregation=flatten`。
- 四分类顺序固定为：`left_hand`、`right_hand`、`feet`、`tongue`。

### 已核验的 50M checkpoint

- 路径：`E:\code\BCI_DayLoop\checkpoints\50M\pretrain_checkpoint_4.pt`
- SHA-256：`97335B696B3AE9138DCB51C736F49EE1C6008FDC22FC42F13EA9A5301452F36E`
- 顶层权重 key：`model_state_dict`
- `time_embed.weight=(10,512)`
- `channel_embed.weight=(64,512)`
- `tokenizer.proj.0.weight=(512,100)`
- objective：基于权重与输出头结构，高置信度为 `timefreq`。

50M Adapter、四分类分类头和模型包加载尚未作为阶段 0.5 交付完成；以上记录不表示它们已接入 Pipeline。

## 环境与命令结果

| 命令 | 结果 |
| --- | --- |
| `git branch --show-current` | `chore/stage-0.5-baseline-contract` |
| `git rev-parse HEAD` | `33a6b19d1fb977461aba23660fb7a39274121e9e` |
| `python --version` | `Python 3.12.9` |
| `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | `2.5.1+cu121 True` |
| `python -m compileall .` | 退出码 0；编译成功。 |
| `pytest` | 退出码 0；11 passed，4 warnings，18.11 秒。 |

## 环境兼容性说明

- 本次实际测试环境为 Python 3.12.9、PyTorch 2.5.1+cu121，CUDA 可用。
- 仓库 `pyproject.toml` 声明 Python `>=3.11,<3.12`，PyTorch `>=2.0.1,<2.1`。
- 因此，本次 `compileall` 和 `pytest` 虽通过，但实际环境不属于项目声明的正式支持范围。
- 此项记录为正式复现前需要处理的环境差异，不视为测试失败。
- 建议正式阶段 0.5 使用项目规定的 Python 3.11 环境重新验证。

## 测试结果

`pytest` 收集 11 项测试，全部通过：

- `tests/test_bnci_windows.py`：1 passed
- `tests/test_data.py`：3 passed
- `tests/test_inference.py`：2 passed
- `tests/test_model.py`：3 passed
- `tests/test_replay.py`：2 passed

警告共 4 条，均来自 MOABB / pyriemann 的弃用提示；未造成测试失败。

## 回放前置检查与缺失项

以下文件/目录均缺失，因此未执行 `python scripts/replay_offline.py --config configs/day1_bnci_s01.yaml --max-windows 20`，且未下载、生成或伪造任何内容：

- `data/processed/bnci2014_001_s01.h5`
- `checkpoints/labram-base.pth`
- `runs/day1_bnci_s01/model_package/`
- `runs/day1_bnci_s01/model_package/model.yaml`
- `runs/day1_bnci_s01/model_package/preprocessing.yaml`
- `runs/day1_bnci_s01/model_package/command_map.json`
- `runs/day1_bnci_s01/model_package/head.pt`

配置 `configs/day1_bnci_s01.yaml` 指向上述 HDF5 数据路径与 checkpoint；离线回放脚本还默认要求 `runs/day1_bnci_s01/model_package/` 作为模型包。

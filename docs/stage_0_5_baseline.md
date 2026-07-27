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
| 窗口长度 | 4 秒 |
| 窗口步长 | 0.5 秒 |

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
